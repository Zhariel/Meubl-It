import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

class ImageManagerWidget extends StatelessWidget {
  final Uint8List? pickedImage;
  final Uint8List? generateImage;
  final Function(DragStartDetails) onPanStart;
  final Function(DragUpdateDetails) onPanUpdate;
  final Function(DragEndDetails) onPanEnd;
  final Rect selectedRect;
  final bool isSelecting;

  const ImageManagerWidget({
    Key? key,
    this.pickedImage,
    this.generateImage,
    required this.onPanStart,
    required this.onPanUpdate,
    required this.onPanEnd,
    required this.selectedRect,
    required this.isSelecting,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Stack(
        children: [
          GestureDetector(
            onPanStart: onPanStart,
            onPanUpdate: onPanUpdate,
            onPanEnd: onPanEnd,
            child: Container(
              child: pickedImage != null || generateImage != null
                  ? generateImage != null
                      ? Image.memory(
                          generateImage!,
                          width: MediaQuery.of(context).size.width,
                          height: MediaQuery.of(context).size.width * (4 / 3),
                        )
                      : Image.memory(
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
      ),
    );
  }

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

Map<String, dynamic> resizeCoordinates(
    double width,
    double height,
    double startXAxis,
    double startYAxis,
    double endXAxis,
    double endYAxis,
    double newWidth,
    double newHeight) {
  final double resizeRatioWidth = newWidth / width;
  final double resizeRatioHeight = newHeight / height;
  final double resizedStartXAxis = startXAxis * resizeRatioWidth;
  final double resizedStartYAxis = startYAxis * resizeRatioHeight;
  final double resizedEndXAxis = endXAxis * resizeRatioWidth;
  final double resizedEndYAxis = endYAxis * resizeRatioHeight;

  return {
    'resizedStartXAxis': resizedStartXAxis,
    'resizedStartYAxis': resizedStartYAxis,
    'resizedEndXAxis': resizedEndXAxis,
    'resizedEndYAxis': resizedEndYAxis
  };
}

Future<Map<String, dynamic>> resizeAndCompressImage(
    Uint8List pickedImage,
    double startXAxis,
    double startYAxis,
    double endXAxis,
    double endYAxis,
    double widthScreen,
    double heightScreen) async {
  final decodedImage = img.decodeImage(pickedImage);
  if (decodedImage != null) {
    final result = resizeCoordinates(widthScreen, heightScreen, startXAxis,
        startYAxis, endXAxis, endYAxis, 500, 500);

    final double resizedStartXAxis = result['resizedStartXAxis'];
    final double resizedStartYAxis = result['resizedStartYAxis'];
    final double resizedEndXAxis = result['resizedEndXAxis'];
    final double resizedEndYAxis = result['resizedEndYAxis'];

    var resizedImage = Uint8List.fromList(
      img.encodePng(img.copyResize(decodedImage, width: 500, height: 500)),
    ).toList();

    return {
      'resizedStartXAxis': resizedStartXAxis,
      'resizedStartYAxis': resizedStartYAxis,
      'resizedEndXAxis': resizedEndXAxis,
      'resizedEndYAxis': resizedEndYAxis,
      'compressedPickedImage': resizedImage,
    };
  } else {
    throw Exception("L'image est null");
  }
}
