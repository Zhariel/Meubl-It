import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

class RequestAPI {
  Future<void> requestGenerateFurnitureAPIService(
      Function(Uint8List?) callback,
      File pickedImage,
      String selectedFurniture,
      double startXAxis,
      double startYAxis,
      double endXAxis,
      double endYAxis) async {
    Uint8List pickedImageBytes = await pickedImage.readAsBytes();
    String base64PickedImage = base64Encode(pickedImageBytes);

    var headers = {'Content-Type': 'application/json'};
    var request = http.Request('POST', Uri.parse(
            'https://fijr8ps8kk.execute-api.us-east-1.amazonaws.com/api_meubl_it/inference_pipeline'
    ));
    request.body = json.encode({
      "encoded_img": base64PickedImage,
      "selected_furniture": selectedFurniture,
      "start-x-axis": startXAxis,
      "start-y-axis": startYAxis,
      "end-x-axis": endXAxis,
      "end-y-axis": endYAxis
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
