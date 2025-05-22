def login(user, password):
    db.connect()
    # ...
    db.close()

def logout(user):
    print("logout")

def get_user(id):
    db.query(f"SELECT * FROM users WHERE id={id}")
    return {} 