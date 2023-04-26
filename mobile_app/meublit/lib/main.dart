import 'dart:io';

import 'package:flutter/material.dart';
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
  bool isPicked = false;
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
                    ? Image.file(
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
                    // Generate picture
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
