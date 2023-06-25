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
          value: 'chair',
          child: Text('Chair'),
        ),
        DropdownMenuItem<String>(
          value: 'bookshelf',
          child: Text('Bookshelf'),
        ),
        DropdownMenuItem<String>(
          value: 'dresser',
          child: Text('Dresser'),
        ),
        DropdownMenuItem<String>(
          value: 'sofa',
          child: Text('Sofa'),
        ),
        DropdownMenuItem<String>(
          value: 'table',
          child: Text('Table'),
        ),
        DropdownMenuItem<String>(
          value: 'no_meuble',
          child: Text("Il n'y a pas de meuble"),
        ),
      ],
    );
  }
}
