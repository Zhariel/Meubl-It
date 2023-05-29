import 'package:flutter/material.dart';
import 'package:meublit/screens/service/image_manager.dart';

class GenerateFurnitureService extends StatefulWidget {
  const GenerateFurnitureService({super.key});

  @override
  _GenerateFurnitureServiceState createState() =>
      _GenerateFurnitureServiceState();
}

class _GenerateFurnitureServiceState extends State<GenerateFurnitureService> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Générateur de Meubles"),
      ),
      body: const Center(
        child: ImageSelector(),
      ),
    );
  }
}
