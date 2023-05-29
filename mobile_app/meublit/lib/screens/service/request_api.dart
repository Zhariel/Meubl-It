import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

class RequestAPI {
  Future<void> requestGenerateFurnitureAPIService(
      Function(Uint8List?) callback,
      File pickedImage,
      Uint8List croppedImage,
      String selectedFurniture) async {
    Uint8List pickedImageBytes = await pickedImage.readAsBytes();
    String base64PickedImage = base64Encode(pickedImageBytes);
    String base64CroppedImage = base64Encode(croppedImage);

    var headers = {'Content-Type': 'application/json'};
    var request = http.Request('POST', Uri.parse(
            'https://3b2l3tcj9j.execute-api.us-east-1.amazonaws.com/dev/api_inference_pipeline'
    ));
    request.body = json.encode({
      "encoded_img": base64PickedImage,
      "encoded_cropped_img": base64CroppedImage,
      "selected_furniture": selectedFurniture
    });
    request.headers.addAll(headers);

    http.StreamedResponse response = await request.send();

    if (response.statusCode == 200) {
      String responseBody = await response.stream.bytesToString();
      Map<String, dynamic> responseJson = jsonDecode(responseBody);
      String base64Image = responseJson['body'];
      Uint8List imageBytes = base64Decode(base64Image);
      callback(imageBytes);
    } else {
      throw Exception(response.reasonPhrase);
    }
  }
}
