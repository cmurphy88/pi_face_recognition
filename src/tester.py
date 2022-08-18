from deepface import DeepFace
import matplotlib.pyplot as pit


def test_face_true():
    img1_path = 'pi_face/images/img1.jpeg'
    img2_path = 'pi_face/images/img2.jpeg'

    img1 = DeepFace.detectFace(img1_path)
    img2 = DeepFace.detectFace(img2_path)

    model_name = 'Facenet'

    resp = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name
    )

    return resp


def test_face_false():
    img1_path = 'pi_face/images/img1.jpeg'
    img3_path = 'pi_face/images/img3.jpeg'

    img1 = DeepFace.detectFace(img1_path)
    img2 = DeepFace.detectFace(img3_path)

    model_name = 'Facenet'

    resp = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img3_path,
        model_name=model_name
    )

    return resp
