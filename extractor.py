import boto3

def detect_text(pic):
    client = boto3.Session(profile_name="default").client('rekognition')
    with open(pic, mode='rb') as imagefile:
        f = imagefile.read()
        buffered = bytearray(f)
        response = client.detect_text(Image={'Bytes': buffered})
        textdetections = response['TextDetections'][0]['DetectedText']
        return textdetections