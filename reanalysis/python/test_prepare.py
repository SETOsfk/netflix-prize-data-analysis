"""pytest reanalysis/python -q — the byte parser is the one piece of logic everything else stands on."""
import numpy as np

from prepare import days, parse, parse_probe


def test_parse_ratings_and_dates():
    buf = np.frombuffer(b"1:\n1488844,3,2005-09-06\n822109,5,2005-05-13\r\n10:\n30878,4,1999-11-11\n", np.uint8)
    u, m, r, d = (np.zeros(8, t) for t in (np.int32, np.int16, np.int8, np.int16))
    n = parse(buf, u, m, r, d, 0)
    assert n == 3 and u[:3].tolist() == [1488844, 822109, 30878] and m[:3].tolist() == [1, 1, 10]
    assert r[:3].tolist() == [3, 5, 4] and d[2] == days(1999, 11, 11)
    assert days(1998, 1, 1) == 0 and days(2005, 12, 31) - days(2005, 1, 1) == 364


def test_parse_probe():
    buf = np.frombuffer(b"1:\n30878\n2647871\n10:\n1952305\n", np.uint8)
    m, u = np.zeros(8, np.int64), np.zeros(8, np.int64)
    k = parse_probe(buf, m, u)
    assert k == 3 and m[:3].tolist() == [1, 1, 10] and u[:3].tolist() == [30878, 2647871, 1952305]
