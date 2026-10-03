"""Demo vehicles. Service text is intentionally messy: mixed date formats, abbreviations, mileage as '45k'."""

NO_GATES = dict(frame_damage=False, flood=False, salvage_title=False, airbag_deployed=False, odometer_mismatch=False)

VEHICLES = {
    "camry": dict(
        label="Clean sedan", vin="4T1B11HK5LU000001", year=2020, make="Toyota", model="Camry", segment="sedan", age=6,
        mileage=71800, base_value=17500, open_recalls=0, collision_events=0, service_tier=3, local_demand=0.92,
        scan=dict(dent_count=2, damage_severity=1.2, broken_parts=0, brake_pad_mm=7.5, oil_degradation=0.15, engine_score=90),
        gates=dict(NO_GATES),
        service_text="""02/10/2021 - 12,300 mi - Oil change & filter - Jiffy Lube
Aug 2021 @ 19k: oil + tire rotation
2022-02-14 25,900 miles: oil chg
Sep 2022 @ 31k - replaced front brake pads and rotors
03/21/2023 - 37,400 mi - oil change, cabin air filter
Oct 2023 @ 43k oil service
2024-04-09 49,100 miles oil chg + tire rotation
Nov 2024 @ 55k: new battery installed
2025-05-12 61,000 mi oil change
Dec 2025 @ 66k - oil service, tire rotation
2026-05-02 71,300 miles oil chg""",
    ),
    "f150": dict(
        label="Damaged pickup", vin="1FTEW1EP7HF000002", year=2017, make="Ford", model="F-150", segment="truck", age=9,
        mileage=118000, base_value=21000, open_recalls=1, collision_events=2, service_tier=2, local_demand=1.0,
        scan=dict(dent_count=11, damage_severity=7.5, broken_parts=3, brake_pad_mm=3.5, oil_degradation=0.5, engine_score=62),
        gates=dict(NO_GATES, frame_damage=True),
        service_text="""2023-11-02 - 96,400 mi - oil change
Mar 2024 @ 101k: replaced front brake pads
06/18/2024 - 104,900 mi - repaired front bumper & fender (collision)
2025-01-22 109,300 miles: oil chg
Aug 2025 @ 113k: control arm replaced, alignment
2026-02-11 117,200 mi - frame inspection - bent rail noted""",
    ),
    "rogue": dict(
        label="Borderline SUV", vin="JN8AT2MV3JW000003", year=2018, make="Nissan", model="Rogue", segment="suv", age=8,
        mileage=96000, base_value=14500, open_recalls=0, collision_events=0, service_tier=2, local_demand=0.84,
        scan=dict(dent_count=4, damage_severity=2.0, broken_parts=1, brake_pad_mm=4.0, oil_degradation=0.35, engine_score=78),
        gates=dict(NO_GATES),
        service_text="""2023-12-05 - 82,000 mi - oil change
Apr 2024 @ 86k - replaced rear brake pads
2024-09-14 89,500 miles: repaired rear bumper (minor collision)
2025-04-30 - 92,100 mi - oil chg + tire rotation
Nov 2025 @ 94k: replaced rear brake pads again""",
    ),
    "civic": dict(
        label="Thin history (Tier 1)", vin="19XFC1F36KE000004", year=2019, make="Honda", model="Civic", segment="hatchback", age=7,
        mileage=64000, base_value=15200, open_recalls=0, collision_events=0, service_tier=1, local_demand=1.05,
        scan=dict(dent_count=3, damage_severity=1.8, broken_parts=0, brake_pad_mm=6.0, oil_degradation=0.3, engine_score=85),
        gates=dict(NO_GATES),
        service_text="""2019-06-10 - 2,100 mi - oil change
2021-02-02 - 30,500 mi - oil change
2023-07-19 - 48,200 mi - oil change
2025-12-01 - 62,100 mi oil chg""",
    ),
    "silverado": dict(
        label="Flood-damaged truck", vin="3GCUKREC1EG000005", year=2014, make="Chevrolet", model="Silverado 1500", segment="truck", age=12,
        mileage=161000, base_value=11000, open_recalls=2, collision_events=0, service_tier=2, local_demand=0.95,
        scan=dict(dent_count=4, damage_severity=2.5, broken_parts=2, brake_pad_mm=4.5, oil_degradation=0.6, engine_score=55),
        gates=dict(NO_GATES, flood=True),
        service_text="""2024-02-10 - 150,300 mi - oil change
Sep 2024 @ 153k - replaced transmission fluid
2025-08-12 - 158,000 mi: oil chg""",
    ),
}
