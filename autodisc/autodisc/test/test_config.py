import autodisc as ad

def test_config():

    config = ad.Config()

    config['a'] = 1
    config.b = 2

    assert config.a == 1
    assert config['a'] == 1

    assert config.b == 2
    assert config['b'] == 2


def test_set_default_values():

    # Simple
    default_val = {'x': 1}

    config1 = ad.Config()
    config1.y = 2
    config1.set_default_values(default_val)

    assert config1.x == 1
    assert config1.y == 2


    # Nested

    default_val = {'x': 1, 'y': {'z': 2}}

    config2 = ad.Config()
    config2.y = ad.Config()
    config2.y.w = 3
    config2.set_default_values(default_val)

    assert config2.x == 1
    assert config2.y.z == 2
    assert config2.y.w == 3


def test_set_default_config():

    # Simple
    default_val = {'x': 1, 'y':-1}

    config1 = ad.Config()
    config1.y = 2

    config1 = ad.config.set_default_config(config1, default_val)

    assert config1.x == 1
    assert config1.y == 2


    # Nested

    default_val = {'x': 1, 'y': {'z': 2}}

    config2 = ad.Config()
    config2.y = ad.Config()
    config2.y.w = 3
    config2 = ad.config.set_default_config(config2, default_val)

    assert config2.x == 1
    assert config2.y.z == 2
    assert config2.y.w == 3



    # several config:

    default_val1 = {'x': 1, 'y': {'z': 2}}
    default_val2 = {'y': {'z': 3}}

    config3 = ad.Config()
    config3.y = ad.Config()
    config3.y.w = 4
    config3 = ad.config.set_default_config(config3, default_val2, default_val1)

    assert config3.x == 1
    assert config3.y.z == 3
    assert config3.y.w == 4



