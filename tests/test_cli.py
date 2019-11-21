
from niimasker import cli


def test_empty_to_none():
    
    result = cli._empty_to_None([])
    assert result is None

    result = cli._empty_to_None(None)
    assert result is None
    
    result = cli._empty_to_None("x")
    assert result == "x"
    

def test_merge_params():

    a = {'cat': 0, 'dog': 1, 'bird': 2}
    b = {'cat': 3, 'dog': 4, 'hamster': 2}

    result = cli._merge_params(a, b)
    expected = {'cat': 3, 'dog': 4, 'hamster': 2, 'bird': 2}
    assert result == expected