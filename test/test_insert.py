from project import db


def insert_data():
    test_image = open_image()
    patients = [
        db.Patients(
            name="John",
            lastname="Doe",
            birth_date="01-01-1994",
            condition="Flu",
            prescriptions=["Paracetamol"],
            original_image=test_image,
        ),
        db.Patients(
            name="Jane",
            lastname="Smith",
            birth_date="06-06-1986",
            condition="Migraine",
            prescriptions=["Ibuprofen"],
            original_image=test_image,
        ),
        db.Patients(
            name="Alice",
            lastname="Brown",
            birth_date="10-10-1960",
            condition="Diabetes",
            prescriptions=["Insulin"],
            original_image=test_image,
        ),
        db.Patients(
            name="Bob",
            lastname="Johnson",
            birth_date="12-12-1992",
            condition="Hypertension",
            prescriptions=["Amlodipine"],
            original_image=test_image,
        ),
    ]
    db_session = db.create_session()
    for i in patients:
        db_session.add(i)
    db_session.commit()


def open_image():
    with open("resources/yes/Y2.jpg", "rb") as file:
        return file.read()


if __name__ == "__main__":
    insert_data()
