# Simplifies generated voxel mesh
#
# Usage:
#
# python simplify.py <in:obj_filename> <out:obj_filename>
#
# Authors
#  - Thomas Krijnen <thomas@aecgeeks.com>
#
# Changelog
#  - 2020 feb 17 - initial version commited to repo
#

from __future__ import print_function

import sys
import numpy
import operator
import itertools

from collections import defaultdict

ifn, ofn = sys.argv[1:]

out = open(ofn, "w")


def vertices():
    with open(ifn) as f:
        for l in f:
            if l.startswith("v "):
                yield tuple(map(float, l.split(" ")[1:]))


verts = numpy.array(list(vertices()))


def groups():
    current = []
    name = None
    with open(ifn) as f:
        for l in f:
            if l.startswith("g "):
                if current:
                    yield name, current
                    current[:] = []
                name = l[2:].strip()
            elif l.startswith("f "):
                idxs = l.split(" ")[1:]
                vidx, nidx = zip(*(s.strip().split("//") for s in idxs))
                assert nidx[0] == nidx[1] and nidx[0] == nidx[2]
                current.append((int(nidx[0]), tuple(map(int, vidx))))
            elif l[0] != 'v':
                # @nb only single mtllib and usemtl supported
                print(l.strip(), file=out)
    if current:
        yield name, current
        current[:] = []


def construct_graph(adjacency_list):
    graph = defaultdict(set)
    for a, b in adjacency_list:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def split_graph_at(vertices, G, breaks):
    raise NotImplementedError("Reimplement without recursion")

    visited = set()
    breaks = set(breaks)

    def visit(v):
        visited.add(v)
        nb = list(G[v] - visited - breaks)
        for n in nb:
            visit(n)

    Vset = set(vertices) - breaks

    while Vset:
        visit(next(iter(Vset)))
        yield numpy.array(list(visited))
        Vset -= visited
        visited.clear()


def decompose_at_break(faces):
    pass


num_vertices = 1

for gn, g in groups():
    print("g", gn, file=out)

    def pos_along_normal(p): return verts[p[1][0]-1][(p[0]-1) // 2]
    def annotate_with_pos_along_normal(t): return (t[0], pos_along_normal(t), t[1])

    g = sorted(map(annotate_with_pos_along_normal, g))

    # print(g)
    for (normal, p_along_n), normal_triangles in itertools.groupby(g, lambda t: (t[0], t[1])):
        faces = numpy.array([t[2] for t in normal_triangles])
        # print(faces)
        edges = numpy.concatenate((
            faces[:, 0:2],
            faces[:, 1:3],
            numpy.concatenate((faces[:, 2:], faces[:, :1]), axis=1)
        ))
        edges.sort(axis=1)
        unique_edges, idxs, inverses, counts = numpy.unique(edges, return_index=True, return_counts=True, return_inverse=True, axis=0)
        borders = unique_edges[counts == 1]

        vertices = numpy.unique(faces)

        V = vertices.size
        E = unique_edges.shape[0]
        F = faces.shape[0]

        # print(V, E, F, V-E+F)
        # @todo check for inner boundaries here?

        esminfs = numpy.unique(unique_edges, return_counts=True)[1] - numpy.unique(faces, return_counts=True)[1]
        break_points = vertices[esminfs == 2]
        if break_points.size:
            zz = defaultdict(list)  # eidx -> faceidx
            zzz = defaultdict(set)  # faceidx -> faceidx
            for i, eidx in enumerate(inverses):
                zz[eidx].append(i % len(faces))
            for fidxs in zz.values():
                if len(fidxs) == 2:
                    a, b = fidxs
                    zzz[a].add(b)
                    zzz[b].add(a)

            face_idxs = set(range(len(faces)))
            visited_faceidxs = set()

            def visit_faceidx(i):
                visited_faceidxs.add(i)
                nb = zzz[i] - visited_faceidxs
                for n in nb:
                    visit_faceidx(n)

            faces_list = []

            while face_idxs:
                visit_faceidx(next(iter(face_idxs)))
                arr = numpy.array(list(visited_faceidxs))
                faces_list.append(faces[arr])
                face_idxs -= visited_faceidxs
                visited_faceidxs.clear()
            # import pdb; pdb.set_trace()
        else:
            faces_list = [faces]

        loops_all = []

        for faces in faces_list:

            loops = []

            # @todo duplicated from above

            edges = numpy.concatenate((
                faces[:, 0:2],
                faces[:, 1:3],
                numpy.concatenate((faces[:, 2:], faces[:, :1]), axis=1)
            ))
            edges.sort(axis=1)
            unique_edges, idxs, inverses, counts = numpy.unique(edges, return_index=True, return_counts=True, return_inverse=True, axis=0)
            borders = unique_edges[counts == 1]

            graph = construct_graph(borders)

            visited = set()

            def visit_loop(G, v):
            
                arr = []
                first = True
            
                while True:
                
                    visited.add(v)
                    # if arr: print(verts[numpy.array(arr)])
                    arr.append(v)
                    
                    nb = list(G[v] - visited)
                    if len(nb) == 0:
                        break
                        
                    assert len(nb) == [1, 2][first]
                    
                    v = nb[0]
                    first = False
                
                return arr

            def visit(G, v):
                # visited.add(v)
                # nb = list(G[v] - visited)
                # for n in nb:
                #     visit(G, n)
                
                queue = [v]
                while queue:
                    # print(*queue)
                    v = queue[0]
                    queue = queue[1:]
                    
                    visited.add(v)
                    nb = list(G[v] - visited - set(queue))
                    queue.extend(nb)

            while len(visited) < len(graph.keys()):
                v = next(iter(set(graph.keys()) - visited))
                loop = numpy.array(visit_loop(graph, v))
                loop -= 1
                loop.shape += (1,)
                loop_shifted = numpy.roll(loop, 1, axis=0)
                deltas = verts[loop_shifted] - verts[loop]
                deltas_shifted = numpy.roll(deltas, -1, axis=0)
                # @nb compare with epsilon
                equal_deltas = numpy.any(numpy.abs((deltas - deltas_shifted)) > 1.e-10, axis=2)
                if len(list(loop[equal_deltas])) == 3 and len(loop) == 4:
                    import pdb; pdb.set_trace()
                loops.append(loop[equal_deltas])

            if len(loops) > 1:
                # We need to find if there are inner loops

                full_graph = construct_graph(unique_edges - 1)
                # print(full_graph)
                
                inner_bounds = set()

                def loop_area(lp):
                    N = (normal-1)// 2
                    retain = list(range(3))
                    retain.remove(N)
                    loop_2d = verts[lp][:,retain]
                    x, y = loop_2d.T
                    return 0.5 * numpy.abs(
                        numpy.dot(x,numpy.roll(y,1)) - \
                        numpy.dot(y,numpy.roll(x,1)))
                        
                loop_areas = list(map(loop_area, loops))
                        
                for ai, bi in itertools.combinations(list(range(len(loops))), 2):
                    a, b = loops[ai], loops[bi]
                    visited.clear()
                    visit(full_graph, a[0])
                    if visited & set(b):
                        # print(verts[faces-1])
                        # print(verts[a])
                        # print(verts[b])
                        # for x in (set(a), visited, set(b)):
                        #     for xx, xxx in zip(x, verts[numpy.array(list(x))]):
                        #         print(xx, *xxx)
                        #     print("--")
                                                
                                                
                        if loop_areas[ai] > loop_areas[bi]:
                            inner_bounds.add(bi)
                        else:
                            inner_bounds.add(ai)

                        # raise ValueError("Inner boundaries not supported yet")
                        
                for b_idx in sorted(inner_bounds, reverse=True):
                    print("Warning, ignored inner boundary with area, ", loop_areas[b_idx])
                    loops[b_idx:b_idx+1] = []

            loops_all.extend(loops)

        for i, loop in enumerate(loops_all):
            # print("g", "%s-%d-%d-%d" % (gn, p_along_n / 0.5, normal, i), file=out)
            for v in verts[loop]:
                print("v", *v, file=out)
            print("f", *(numpy.arange(len(loop), dtype=int) + num_vertices), file=out)
            num_vertices += len(loop)
