import sys, xml.etree.ElementTree as ET
for f in sys.argv[1:]:
    for tc in ET.parse(f).iter("testcase"):
        fl = tc.find("failure")
        msg = "PASSED" if fl is None else fl.get("message", "").splitlines()[0][:150]
        print(f"{tc.get('classname').split('.')[-1].removeprefix('test_m68c_')}::{tc.get('name')} -> {msg}")
