import 'package:flutter/material.dart';
import 'package:meublit/screens/service.dart';

void main() {
  runApp(const MeublItApp());
}

class MeublItApp extends StatelessWidget {
  const MeublItApp({Key? key}) : super(key: key);

  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Meubl-It',
      theme: ThemeData(
        primarySwatch: Colors.deepOrange,
      ),
      home: const GenerateFurnitureService(),
      debugShowCheckedModeBanner: false,
    );
  }
}
