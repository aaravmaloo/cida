from app.core.security import hash_passkey


if __name__ == "__main__":
    import getpass

    value = getpass.getpass("Admin passkey: ")
    print(hash_passkey(value))

