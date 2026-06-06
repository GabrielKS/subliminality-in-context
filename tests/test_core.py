from subliminality import greet


def test_greet_includes_name():
    assert "Ada" in greet("Ada")


def test_greet_is_a_string():
    assert isinstance(greet("world"), str)
