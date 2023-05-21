import sys

import numpy as np

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement
from ifcopenshell.util.unit import calculate_unit_scale

MAX_FACES = 100
THICKNESS = 0.05
EPSILON = 1.e-5

s = ifcopenshell.geom.settings(USE_WORLD_COORDS=True, DISABLE_OPENING_SUBTRACTIONS=True)
s2 = ifcopenshell.geom.settings(USE_WORLD_COORDS=True, DISABLE_OPENING_SUBTRACTIONS=True, CONVERT_BACK_UNITS=True)

def get_cartesiantransformationoperator(inst):
    origin = np.array(inst.LocalOrigin.Coordinates)
    axis1 = np.array((1., 0., 0.))
    axis2 = np.array((0., 1., 0.))
    axis3 = np.array((0., 0., 1.))

    if inst.Axis1:
        axis1[0:3] = inst.Axis1.DirectionRatios
    if inst.Axis2:
        axis2[0:3] = inst.Axis2.DirectionRatios
    if inst.Axis3:
        axis3[0:3] = inst.Axis3.DirectionRatios

    m4 = ifcopenshell.util.placement.a2p(origin, axis3, axis1)
    # Negate axis2 (introduce mirroring) when supplied axis2
    # is opposite of constructed axis2 from placement
    if m4.T[1][0:3].dot(axis2) < 0.:
        m4.T[1] *= -1.

    scale1 = scale2 = scale3 = 1.

    if inst.Scale:
        scale1 = inst.Scale

    if inst.is_a('IfcCartesianTransformationOperator3DnonUniform'):
        scale2 = inst.Scale2 if inst.Scale2 is not None else scale1
        scale3 = inst.Scale3 if inst.Scale3 is not None else scale1

    m4.T[0] *= scale1
    m4.T[1] *= scale2
    m4.T[2] *= scale3

    return m4

def traverse(el, mat=None, visited_face=False):
    if el.is_a('IfcCartesianPoint'):
        if visited_face:
            p = np.array(el.Coordinates + (1,))
            if mat is not None:
                p = mat @ p
            yield p
        return

    elif el.is_a('IfcExtrudedAreaSolid'):
        try:
            shp = ifcopenshell.geom.create_shape(s2, el)
        except:
            return
        for p in np.array(shp.verts).reshape((-1, 3)):
            p = np.concatenate((p, [1.]))
            if mat is not None:
                p = mat @ p
            yield p
        return

    new_mat = None

    if el.is_a('IfcMappedItem'):
        new_mat = ifcopenshell.util.placement.get_axis2placement(el.MappingSource.MappingOrigin)
        new_mat = get_cartesiantransformationoperator(el.MappingTarget) @ new_mat
        if mat is not None:
            new_mat = mat @ new_mat

    for x in el.wrapped_data.file.traverse(el, 1)[1:]:
        yield from traverse(x, mat=new_mat if new_mat is not None else mat, visited_face=visited_face or el.is_a('IfcFace'))

#@todo there are nans in ys

fn1, fn2 = sys.argv[1:]

def do_try(fn, default=None):
    try:
        return fn()
    except:
        return default

f = ifcopenshell.open(fn1)

factor = calculate_unit_scale(f)

to_remove = []

for prod in f.by_type('IfcDoor') + f.by_type('IfcWindow'):
    print(prod)
    num_faces = sum(1 for inst in f.traverse(prod) if inst.is_a('IfcFace'))
    
    if num_faces < MAX_FACES:
        print(f"Not too many faces {num_faces} < {MAX_FACES}")
        continue

    body = [r for r in prod.Representation.Representations if r.RepresentationIdentifier == 'Body']
    if not body:
        print(f"No body representation")
        continue
    body = body[0]

    opening = do_try(lambda: prod.FillsVoids[0].RelatingOpeningElement)
    if not opening:
        print(f"No opening")
        vs = list(traverse(prod))
        if not vs:
            print("No points, continue")
            continue
        
        vs = np.array(vs)
        minx, miny, minz = map(float, np.amin(vs, axis=0)[0:3])
        maxx, maxy, maxz = map(float, np.amax(vs, axis=0)[0:3])
        
        to_remove.append(prod.Representation)

        prod.Representation = f.createIfcProductDefinitionShape(
            None, None,
            [f.createIfcShapeRepresentation(
                body.ContextOfItems,
                'Body',
                'SweptSolid',
                [f.createIfcExtrudedAreaSolid(
                    f.createIfcRectangleProfileDef(
                        'AREA', None,
                        f.createIfcAxis2Placement2D(
                            f.createIfcCartesianPoint((
                                -((maxx + minx) / 2.),
                                0.,
                            ))
                        ),
                        maxx - minx,
                        THICKNESS / factor
                    ),
                    f.createIfcAxis2Placement3D(
                        f.createIfcCartesianPoint((0., 0., minz))
                    ),
                    f.createIfcDirection((0., 0., 1.)),
                    maxz - minz
                )]
            )]
        )
        
        continue

    wall = do_try(lambda: opening.VoidsElements[0].RelatingBuildingElement)
    if not (wall and wall.is_a('IfcWall')):
        print(f"No wall")
        continue
    
    print("Replacing...")

    wall_shapes = []
    try:
        ws = ifcopenshell.geom.create_shape(s, wall)
        wall_shapes.append(ws)
    except:
        if wall.IsDecomposedBy:
            for elem in wall.IsDecomposedBy[0].RelatedObjects:
                ws = ifcopenshell.geom.create_shape(s, elem)
                wall_shapes.append(ws)
                
    if not wall_shapes:
        print(f"No wall geom")
        continue
        
    opening_shape = ifcopenshell.geom.create_shape(s, opening)
    
    m4_door = ifcopenshell.util.placement.get_local_placement(prod.ObjectPlacement)
    m4_door[0:3, 3] *= factor

    ys = []
    
    for wall_shape in wall_shapes:

        vs = wall_shape.geometry.verts
        vs = np.array([vs[i:i+3] for i in range(0, len(vs), 3)])
        vs = np.concatenate((vs, np.ones((vs.shape[0],1))), axis=1)

        vs_local = (np.asmatrix(np.linalg.inv(m4_door)) * np.asmatrix(vs).T).A.T

        ed = wall_shape.geometry.edges
        ed = np.array([ed[i:i+2] for i in range(0, len(ed), 2)])

        for ab in vs_local[ed]:
            d = np.diff(ab, axis=0).flatten()
            if abs(d[2]) > EPSILON:
                continue

            if sum((d*d)[0:3]) < EPSILON:
                continue

            if abs(d[1]) > abs(d[0]):
                continue

            if not (len(set(np.sign(ab[:,0]))) == 2 or np.min(np.abs(ab[:,0])) < EPSILON):
                continue

            (ax, ay), (bx, by) = ab[:,0:2]
            y = ay + (-ax / (bx - ax)) * (by - ay)

            ys.append(float(y))

    if np.any(np.isnan(ys)) or len(ys) < 2:
        print(f"No local wall thickness measured, taking global thickness", *ys)
        ys = list(map(float, vs_local[:, 1]))

    miny, maxy = min(ys), max(ys)

    vs = opening_shape.geometry.verts
    vs = np.array([vs[i:i+3] for i in range(0, len(vs), 3)])
    vs = np.concatenate((vs, np.ones((vs.shape[0],1))), axis=1)

    vs_local = (np.asmatrix(np.linalg.inv(m4_door)) * np.asmatrix(vs).T).A.T

    minx, maxx = map(float, (np.min(vs_local[:,0]), np.max(vs_local[:,0])))
    minz, maxz = map(float, (np.min(vs_local[:,2]), np.max(vs_local[:,2])))

    to_remove.append(prod.Representation)

    prod.Representation = f.createIfcProductDefinitionShape(
        None, None,
        [f.createIfcShapeRepresentation(
            body.ContextOfItems,
            'Body',
            'SweptSolid',
            [f.createIfcExtrudedAreaSolid(
                f.createIfcRectangleProfileDef(
                    'AREA', None,
                    f.createIfcAxis2Placement2D(
                        f.createIfcCartesianPoint((
                            ((maxx - minx) / 2.) / factor,
                            ((maxy + miny) / 2.) / factor,
                        ))
                    ),
                    (maxx - minx) / factor,
                    THICKNESS / factor
                ),
                f.createIfcAxis2Placement3D(
                    f.createIfcCartesianPoint((0., 0., minz / factor))
                ),
                f.createIfcDirection((0., 0., 1.)),
                (maxz - minz) / factor
            )]
        )]
    )

f.write(fn2)