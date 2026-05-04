# Decisiones de arquitectura (Fase 1)

- **Stem CIFAR:** la ResNet estándar ImageNet usa conv 7×7 con stride 2 y maxpool; para entradas 32×32 se sustituye la primera convolución por 3×3, stride 1, padding 1, y `maxpool` por identidad para conservar resolución útil en bloques profundos.
- **Módulos monitoreados:** `layer2` y `layer3` se declaran como foco para futuras penalizaciones de activación y análisis de modularidad (profundidad intermedia).
