import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:meublit/screens/features/furniture_selector.dart';
import 'package:meublit/screens/features/image_manager.dart';
import 'package:meublit/screens/features/request_api.dart';

class FurnitureGenerator extends StatefulWidget {
  const FurnitureGenerator({Key? key}) : super(key: key);

  @override
  State<FurnitureGenerator> createState() => _FurnitureGeneratorState();
}

class _FurnitureGeneratorState extends State<FurnitureGenerator> {
  final RequestAPI _requestAPI = RequestAPI();
  Uint8List? pickedImage;
  Uint8List? generateImage;
  bool isGenerate = false;

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

  Future<void> _pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      final File pickedFile = File(image.path);
      pickedImage = await pickedFile.readAsBytes();

      setState(() {
        generateImage = null;
        isGenerate = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        ImageManagerWidget(
          pickedImage: pickedImage,
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
            callback: (dropdownValue) =>
                setState(() => selectedFurniture = dropdownValue),
          ),
        ),
        const SizedBox(
          height: 10.0,
        ),
        Visibility(
          visible: !isGenerate &&
              selectedFurniture != null &&
              isSelecting &&
              pickedImage != null,
          child: ElevatedButton(
            onPressed: () async {
              if (pickedImage != null) {
                _requestAPI.requestGenerateFurnitureAPIService(
                    updateFromAPI,
                    pickedImage!,
                    selectedFurniture!,
                    startPoint.dx,
                    startPoint.dy,
                    endPoint.dx,
                    endPoint.dy);
              }
              setState(() {
                isSelecting = false;
                isGenerate = true;
                pickedImage = null;
                selectedFurniture = null;
              });
            },
            child: const Text("Générer le meuble"),
          ),
        ),
        Visibility(
          visible: !isGenerate &&
              selectedFurniture != null &&
              isSelecting &&
              pickedImage != null,
          child: const SizedBox(
            height: 10.0,
          ),
        ),
      ],
    );
  }
}
