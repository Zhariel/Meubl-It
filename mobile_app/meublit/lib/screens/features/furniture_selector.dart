import 'package:flutter/material.dart';

class FurnitureSelector extends StatefulWidget {
  final Function(String?) callback;

  const FurnitureSelector({super.key, required this.callback});

  @override
  _FurnitureSelectorState createState() => _FurnitureSelectorState();
}

class _FurnitureSelectorState extends State<FurnitureSelector> {
  String? selectedFurniture;

  @override
  Widget build(BuildContext context) {
    return DropdownButton<String>(
      value: selectedFurniture,
      hint: const Text('Sélectionnez un meuble'),
      onChanged: (newFurniture) {
        widget.callback(newFurniture);
        setState(() {
          selectedFurniture = newFurniture;
        });
      },
      items: const [
        DropdownMenuItem<String>(
          value: 'Meuble1',
          child: Text('Meuble 1'),
        ),
        DropdownMenuItem<String>(
          value: 'Meuble2',
          child: Text('Meuble 2'),
        ),
        DropdownMenuItem<String>(
          value: 'Meuble3',
          child: Text('Meuble 3'),
        ),
        DropdownMenuItem<String>(
          value: 'Meuble4',
          child: Text('Meuble 4'),
        ),
      ],
    );
  }
}
