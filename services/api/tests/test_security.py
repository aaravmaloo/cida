from app.core.security import create_admin_token


def test_admin_token_creation():
    token, expires = create_admin_token()
    assert isinstance(token, str)
    assert expires is not None

