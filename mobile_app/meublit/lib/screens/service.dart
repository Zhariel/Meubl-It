import 'package:flutter/material.dart';
import 'package:meublit/screens/features/captcha.dart';
import 'package:meublit/screens/features/furniture_generator.dart';

class GenerateFurnitureService extends StatefulWidget {
  const GenerateFurnitureService({super.key});

  @override
  State<GenerateFurnitureService> createState() =>
      _GenerateFurnitureServiceState();
}

class _GenerateFurnitureServiceState extends State<GenerateFurnitureService> {
  final List<Widget> _widgets = [];
  int _indexSelected = 0;

  @override
  void initState() {
    super.initState();

    _widgets.addAll([
      Captcha(onChangedStep: (index) => setState(() => _indexSelected = index)),
      Scaffold(
        appBar: AppBar(
          title: const Text("Générateur de Meubles"),
        ),
        body: const Center(
          child: FurnitureGenerator(),
        ),
      )
    ]);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      child: _widgets.isEmpty
          ? const SafeArea(
              child: Scaffold(
                body: Center(
                  child: Text('Loading...'),
                ),
              ),
            )
          : _widgets.elementAt(_indexSelected),
    );
  }
}
