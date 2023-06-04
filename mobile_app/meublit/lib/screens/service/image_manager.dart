import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:meublit/screens/service/furniture_selector.dart';
import 'package:meublit/screens/service/request_api.dart';

class ImageSelector extends StatefulWidget {
  const ImageSelector({Key? key}) : super(key: key);

  @override
  _ImageSelectorState createState() => _ImageSelectorState();
}

class _SelectionPainter extends CustomPainter {
  final Rect selectedRect;
  final bool isSelecting;

  _SelectionPainter(this.selectedRect, this.isSelecting);

  @override
  void paint(Canvas canvas, Size size) {
    if (isSelecting) {
      final paint = Paint()
        ..color = Colors.blue
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;

      canvas.drawRect(selectedRect, paint);
    }
  }

  @override
  bool shouldRepaint(_SelectionPainter oldDelegate) {
    return selectedRect != oldDelegate.selectedRect ||
        isSelecting != oldDelegate.isSelecting;
  }
}

class _ImageSelectorState extends State<ImageSelector> {
  final RequestAPI _requestAPI = RequestAPI();
  File? pickedImage;
  Uint8List? croppedImage;
  Uint8List? generateImage;

  Offset startPoint = Offset.zero;
  Offset endPoint = Offset.zero;
  Rect selectedRect = Rect.zero;
  bool isSelecting = false;

  String? selectedFurniture;

  void updateFromAPI(Uint8List? responseImage) {
    setState(() {
      generateImage = responseImage;
    });
  }

  void updateDropdownButton(String? dropdownValue) {
    setState(() {
      selectedFurniture = dropdownValue;
    });
  }

  img.Image removeSelectedArea(img.Image originalImage, int x, int y, int width,
      int height) {
    img.Image modifiedImage = img.copyResize(originalImage,
        width: originalImage.width, height: originalImage.height);

    for (int i = x; i < x + width; i++) {
      for (int j = y; j < y + height; j++) {
        modifiedImage.setPixel(i, j, img.ColorRgba8(0, 0, 0, 0));
      }
    }

    return modifiedImage;
  }

  Future<void> _cropImage(File image) async {
    int x = selectedRect.left.toInt();
    int y = selectedRect.top.toInt();
    int width = selectedRect.width.toInt();
    int height = selectedRect.height.toInt();

    img.Image? theImage = await img.decodeImageFile(image.path);

    int imageWidth = theImage!.width;
    int imageHeight = theImage.height;
    x = ((selectedRect.left) *
        (imageWidth / MediaQuery.of(context).size.width)).toInt();
    y = ((selectedRect.top) *
        (imageHeight / (MediaQuery.of(context).size.width * (4 / 3)))).toInt();
    width = ((selectedRect.width) *
        (imageWidth / MediaQuery.of(context).size.width)).toInt();
    height = ((selectedRect.height) *
        (imageHeight / (MediaQuery.of(context).size.width * (4 / 3)))).toInt();

    img.Image modifiedImage = removeSelectedArea(theImage, x, y, width, height);
    Uint8List modifiedImageBytes = img.encodeJpg(modifiedImage);

    setState(() {
      croppedImage = modifiedImageBytes;
      isSelecting = false;
    });
  }

  Future<void> _pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      pickedImage = File(image.path);
      setState(() {
        generateImage = null;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Expanded(
            child: Stack(
              children: [
                GestureDetector(
                  onPanStart: (DragStartDetails details) {
                    setState(() {
                      startPoint = details.localPosition;
                      isSelecting = true;
                    });
                  },
                  onPanUpdate: (DragUpdateDetails details) {
                    setState(() {
                      endPoint = details.localPosition;
                      selectedRect = Rect.fromPoints(startPoint, endPoint);
                    });
                  },
                  onPanEnd: (DragEndDetails details) {},
                  child: Container(
                    child: pickedImage != null || generateImage != null
                        ? generateImage != null
                        ? Image.memory(
                      generateImage!,
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
                CustomPaint(
                  painter: _SelectionPainter(selectedRect, isSelecting),
                ),
              ],
            )),
        const SizedBox(
          height: 10.0,
        ),
        ElevatedButton(
          onPressed: () async {
            _pickImage();
          },
          child: const Text("Sélectionner une Image"),
        ),
        const SizedBox(
          height: 10.0,
        ),
        Center(
          child: FurnitureSelector(
            callback: (dropdownValue) => setState(
                    () => selectedFurniture = dropdownValue
            ),
          ),
        ),
        const SizedBox(
          height: 10.0,
        ),
        ElevatedButton(
          onPressed: () async {
            if (pickedImage != null) {
              await _cropImage(pickedImage!);
              _requestAPI.requestGenerateFurnitureAPIService(
                  updateFromAPI,
                  pickedImage!,
                  selectedFurniture!,
                  startPoint.dx,
                  startPoint.dy,
                  endPoint.dx,
                  endPoint.dy
              );
            }
          },
          child: const Text("Générer le meuble"),
        ),
      ],
    );
  }
}
