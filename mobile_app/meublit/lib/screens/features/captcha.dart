import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:meublit/screens/features/image_manager.dart';
import 'package:meublit/screens/features/furniture_selector.dart';
import 'package:meublit/screens/features/request_api.dart';

class Captcha extends StatefulWidget {
  final Function(int) onChangedStep;

  const Captcha({Key? key, required this.onChangedStep}) : super(key: key);

  @override
  State<Captcha> createState() => _CaptchaState();
}

class _CaptchaState extends State<Captcha> {
  final RequestAPI _requestAPI = RequestAPI();
  Uint8List? captchaImage;
  String? keyImgCaptcha;
  bool captchaValid = false;
  Uint8List? generateImage;

  Offset startPoint = Offset.zero;
  Offset endPoint = Offset.zero;
  Rect selectedRect = Rect.zero;
  bool isSelecting = false;

  String? selectedFurniture;

  void getCaptchaImage(Uint8List? responseImage, String keyImg) {
    setState(() {
      captchaImage = responseImage;
      keyImgCaptcha = keyImg;
    });
  }

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

  @override
  void initState() {
    super.initState();
    _requestAPI.getCaptchaFromAPI(getCaptchaImage);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Captcha"),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            ImageManagerWidget(
              pickedImage: captchaImage,
              generateImage: generateImage,
              onPanStart: (details) {
                setState(() {
                  startPoint = details.localPosition;
                  isSelecting = true;
                });
              },
              onPanUpdate: (details) {
                setState(() {
                  endPoint = details.localPosition;
                  final double width = endPoint.dx - startPoint.dx;
                  final double height = endPoint.dy - startPoint.dy;
                  double sign = 1.0;
                  if (width.sign != height.sign) {
                    sign = -1.0;
                  }
                  if (width.abs() < height.abs()) {
                    endPoint = startPoint + Offset(width, width * sign);
                  } else {
                    endPoint = startPoint + Offset(height * sign, height);
                  }
                  selectedRect = Rect.fromPoints(startPoint, endPoint);
                });
              },
              onPanEnd: (details) {},
              selectedRect: selectedRect,
              isSelecting: isSelecting,
            ),
            const SizedBox(
              height: 10.0,
            ),
            Text("Veuillez sélectionner le meuble"),
            const SizedBox(
              height: 10.0,
            ),
            Center(
              child: FurnitureSelector(
                callback: (dropdownValue) =>
                    setState(() => selectedFurniture = dropdownValue),
              ),
            ),
            const SizedBox(
              height: 10.0,
            ),
            Visibility(
              visible: selectedFurniture != null &&
                  isSelecting &&
                  captchaImage != null,
              child: ElevatedButton(
                onPressed: () async {
                  if (captchaImage != null) {
                    captchaValid = await _requestAPI.requestCaptchaAPIService(
                      keyImgCaptcha!,
                      selectedFurniture!,
                      startPoint.dx,
                      startPoint.dy,
                      endPoint.dx,
                      endPoint.dy,
                    );
                    if (captchaValid) {
                      widget.onChangedStep(1);
                    }
                  }
                },
                child: const Text("Valider le captcha"),
              ),
            ),
            Visibility(
              visible: selectedFurniture != null &&
                  isSelecting &&
                  captchaImage != null,
              child: const SizedBox(
                height: 10.0,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
