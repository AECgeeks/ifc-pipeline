prefix_mapping = {
    "EXA"  : 1.e18,
    "PETA" : 1.e15,
    "TERA" : 1.e12,
    "GIGA" : 1.e9,
    "MEGA" : 1.e6,
    "KILO" : 1.e3,
    "HECTO": 1.e2,
    "DECA" : 1.,
    "DECI" : 1.e-1,
    "CENTI": 1.e-2,
    "MILLI": 1.e-3,
    "MICRO": 1.e-6,
    "NANO" : 1.e-9,
    "PICO" : 1.e-12,
    "FEMTO": 1.e-15,
    "ATTO" : 1.e-18
}

def get_unit(f, unit_type, default=None):
    us = [u for u in f.by_type("IfcUnitAssignment")[0][0] if hasattr(u, "UnitType") and u.UnitType == unit_type]
    if len(us) == 0:
        return default
        
    u = us[0]
                
    F = 1.
    if u.is_a("IfcConversionBasedUnit"):
        fc, u = u.ConversionFactor
        F = fc.wrappedValue
        
    assert u.is_a("IfcSIUnit")
    S = prefix_mapping.get(u.Prefix, 1.)

    return F * S
