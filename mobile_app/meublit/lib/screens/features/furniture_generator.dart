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

  double widthScreen = 0.0;
  double heightScreen = 0.0;

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
              widthScreen = MediaQuery.of(context).size.width;
              heightScreen = MediaQuery.of(context).size.width * (4 / 3);
              startPoint = details.localPosition;
              isSelecting = true;
            });
          },
          onPanUpdate: (details) {
            setState(() {
              endPoint = details.localPosition;
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
        const Text(
            "Veuillez sélectionner un meuble dans le dropdown et la zone où vous voulez le mettre dans l'image",
            textAlign: TextAlign.center),
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
                    endPoint.dy,
                    widthScreen,
                    heightScreen);
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
