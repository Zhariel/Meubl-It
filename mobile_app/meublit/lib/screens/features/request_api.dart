import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:meublit/screens/features/image_manager.dart';

class RequestAPI {
  Future<void> requestGenerateFurnitureAPIService(
      Function(Uint8List?) callback,
      Uint8List pickedImage,
      String selectedFurniture,
      double startXAxis,
      double startYAxis,
      double endXAxis,
      double endYAxis,
      double widthScreen,
      double heightScreen) async {
    final result = await resizeAndCompressImage(pickedImage, startXAxis,
        startYAxis, endXAxis, endYAxis, widthScreen, heightScreen);

    final double resizedStartXAxis = result['resizedStartXAxis'];
    final double resizedStartYAxis = result['resizedStartYAxis'];
    final double resizedEndXAxis = result['resizedEndXAxis'];
    final double resizedEndYAxis = result['resizedEndYAxis'];
    List<int> compressedPickedImage = result['compressedPickedImage'];

    compressedPickedImage = zlib.encode(compressedPickedImage);
    String base64PickedImage = base64Encode(compressedPickedImage);

    var headers = {'Content-Type': 'application/json'};
    var request = http.Request(
        'POST',
        Uri.parse(
            'https://640g1w60tg.execute-api.us-east-1.amazonaws.com/api_meubl_it/inference_pipeline'));

    request.body = json.encode({
      "encoded_img": base64PickedImage,
      "selected_furniture": selectedFurniture,
      "start-x-axis": resizedStartXAxis,
      "start-y-axis": resizedStartYAxis,
      "end-x-axis": resizedEndXAxis,
      "end-y-axis": resizedEndYAxis
    });
    request.headers.addAll(headers);
    http.StreamedResponse response = await request.send();

    if (response.statusCode == 200) {
      String responseBody = await response.stream.bytesToString();
      Map<String, dynamic> responseJson = jsonDecode(responseBody);
      String base64Image = responseJson['body'];
      Uint8List imageBytesEncoded = base64Decode(base64Image);
      Uint8List imageBytes = Uint8List.fromList(zlib.decode(imageBytesEncoded));
      callback(imageBytes);
    } else {
      throw Exception(response.reasonPhrase);
    }
  }

  Future<bool> requestCaptchaAPIService(
      String keyImgCaptcha,
      String selectedFurniture,
      double startXAxis,
      double startYAxis,
      double endXAxis,
      double endYAxis,
      double widthScreen,
      double heightScreen) async {
    final result = resizeCoordinates(widthScreen, heightScreen,
        startXAxis, startYAxis, endXAxis, endYAxis, 500, 500);

    final double resizedStartXAxis = result['resizedStartXAxis'];
    final double resizedStartYAxis = result['resizedStartYAxis'];
    final double resizedEndXAxis = result['resizedEndXAxis'];
    final double resizedEndYAxis = result['resizedEndYAxis'];

    var headers = {'Content-Type': 'application/json'};
    var request = http.Request(
        'POST',
        Uri.parse(
            'https://640g1w60tg.execute-api.us-east-1.amazonaws.com/api_meubl_it/valid_captcha'));

    request.body = json.encode({
      "key_img_captcha": keyImgCaptcha,
      "selected_furniture": selectedFurniture,
      "start-x-axis": resizedStartXAxis,
      "start-y-axis": resizedStartYAxis,
      "end-x-axis": resizedEndXAxis,
      "end-y-axis": resizedEndYAxis
    });
    request.headers.addAll(headers);
    http.StreamedResponse response = await request.send();

    if (response.statusCode == 200) {
      String responseBody = await response.stream.bytesToString();
      Map<String, dynamic> responseJson = jsonDecode(responseBody);
      bool responseCaptcha = bool.parse(responseJson['body']);
      return responseCaptcha;
    } else {
      throw Exception(response.reasonPhrase);
    }
  }

  Future<void> getCaptchaFromAPI(Function(Uint8List?, String) callback) async {
    var headers = {'Content-Type': 'application/json'};
    var request = http.Request(
        'POST',
        Uri.parse(
            'https://640g1w60tg.execute-api.us-east-1.amazonaws.com/api_meubl_it/get_captcha'));

    request.headers.addAll(headers);
    http.StreamedResponse response = await request.send();

    if (response.statusCode == 200) {
      String responseBody = await response.stream.bytesToString();
      Map<String, dynamic> responseJson = jsonDecode(responseBody);
      String base64Image = responseJson['body']['img_captcha'];
      String keyImage = responseJson['body']['key_img_captcha'];
      Uint8List imageBytesEncoded = base64Decode(base64Image);
      Uint8List imageBytes = Uint8List.fromList(zlib.decode(imageBytesEncoded));
      callback(imageBytes, keyImage);
    } else {
      throw Exception(response.reasonPhrase);
    }
  }
}
