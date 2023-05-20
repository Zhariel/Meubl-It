import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.@override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'All About Flutter',
      theme: ThemeData(
        primarySwatch: Colors.deepOrange,
      ),
      home: const ImagePickerTutorial(),
    );
  }
}

class ImagePickerTutorial extends StatefulWidget {
  const ImagePickerTutorial({Key? key}) : super(key: key);

  @override
  _ImagePickerTutorialState createState() => _ImagePickerTutorialState();
}

class _ImagePickerTutorialState extends State<ImagePickerTutorial> {
  File? pickedImage;
  Uint8List? responseImage;
  bool isPicked = false;
  bool isResponse = false;
  String imageUrl = '';

  Future<void> _uploadImage(File image) async {
    List<int> imageBytes = await image.readAsBytes();
    String base64Image = base64Encode(imageBytes);

    var headers = {'Content-Type': 'application/json'};
    var request = http.Request(
        'POST',
        Uri.parse(
            'https://3b2l3tcj9j.execute-api.us-east-1.amazonaws.com/dev/api_inference_pipeline'));
    request.body = json.encode({"encoded_img": base64Image});
    request.headers.addAll(headers);

    http.StreamedResponse response = await request.send();

    if (response.statusCode == 200) {
      String responseBody = await response.stream.bytesToString();
      Map<String, dynamic> responseJson = jsonDecode(responseBody);
      String base64Image = responseJson['body'];
      print(base64Image);
      Uint8List imageBytes = base64Decode(base64Image);
      print(imageBytes);
      setState(() {
        responseImage = imageBytes;
        isPicked = true;
        isResponse = true;
      });
    } else {
      print(response.reasonPhrase);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Image Picker"),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Expanded(
              child: Container(
                child: isPicked
                    ? isResponse
                        ? Image.memory(
                            responseImage!,
                            width: MediaQuery.of(context).size.width,
                            height: MediaQuery.of(context).size.width * (4 / 3),
                          )
                        : Image.file(
                            pickedImage!,
                            width: MediaQuery.of(context).size.width,
                            height: MediaQuery.of(context).size.width * (4 / 3),
                          )
                    : Container(
                        color: Colors.blueGrey[100],
                        width: MediaQuery.of(context).size.width,
                        height: MediaQuery.of(context).size.width * (4 / 3),
                      ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(48.0),
              child: ElevatedButton(
                onPressed: () async {
                  final ImagePicker _picker = ImagePicker();
                  final XFile? image =
                      await _picker.pickImage(source: ImageSource.gallery);
                  if (image != null) {
                    pickedImage = File(image.path);
                    setState(() {
                      isPicked = true;
                    });
                  }
                },
                child: const Text("Select Image"),
              ),
            ),
            IgnorePointer(
              ignoring: !isPicked,
              child: Padding(
                padding: const EdgeInsets.all(48.0),
                child: ElevatedButton(
                  onPressed: () {
                    if (pickedImage != null) {
                      _uploadImage(pickedImage!);
                    }
                  },
                  child: const Text("Generate"),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
