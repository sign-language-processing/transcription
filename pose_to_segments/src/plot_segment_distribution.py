import matplotlib.pyplot as plt

segments_default_length = [1.44, 4.0, 6.16, 0.8, 0.48, 1.0, 0.4, 0.16, 0.52, 0.72, 2.32, 1.12, 1.2, 4.12, 3.0, 1.76, 0.32, 1.76, 3.84, 1.48, 0.64, 2.4, 1.12, 0.76, 0.96, 0.6, 0.08, 0.96, 1.44, 0.6, 1.2, 1.6, 6.12, 1.4, 0.08, 1.36, 0.4, 0.56, 0.32, 0.6, 2.64, 1.0, 0.48, 2.2, 1.56, 0.6, 0.96, 0.32, 2.72, 2.8, 0.08, 0.32, 1.64, 0.8, 0.88, 0.6, 0.52, 0.48, 2.96, 0.28, 0.04, 2.24, 1.6, 2.44, 1.08, 0.24, 0.64, 1.28, 0.08, 0.12, 2.92, 2.64, 2.24, 1.04, 0.96, 0.76, 0.2, 1.36, 1.0, 1.28, 3.28, 1.08, 2.04, 0.24, 2.64, 1.8, 1.12, 0.56, 0.08, 2.52, 1.36, 0.08, 3.48, 0.48, 1.56, 3.2, 1.2, 2.44, 1.36, 0.6, 0.72, 1.12, 1.48, 2.32, 0.28, 1.12, 0.16, 0.64, 2.32, 0.48, 0.64, 2.64, 1.64, 0.32, 2.4, 1.0, 0.2, 0.04, 1.16, 2.24, 0.32, 1.6, 4.56, 0.68, 2.0, 1.04, 1.32, 4.72, 3.36, 4.44, 0.6, 2.32, 0.24, 1.32, 0.44, 2.08, 2.48, 0.4, 0.64, 0.72, 2.48, 0.12, 2.76, 0.96, 1.64, 0.88, 1.8, 0.64, 2.0, 2.72, 2.36, 3.6, 0.68, 0.68, 1.16, 0.48, 0.68, 1.44, 0.08, 2.2, 0.04, 1.72, 0.04, 0.2, 0.72, 2.08, 1.6, 0.28, 1.2, 0.4, 1.96, 0.52, 0.72, 0.24, 1.76, 2.24, 2.24, 1.84, 0.08, 4.2, 0.76, 0.04, 2.24, 2.12, 0.2, 2.52, 1.24, 1.16, 0.08, 0.12, 2.12, 1.76, 1.24, 2.84, 0.36, 2.0, 1.04, 5.36, 0.44, 2.6, 0.28, 0.28, 1.72, 0.48, 0.36, 1.44, 1.32, 1.36, 0.92, 0.32, 2.88, 2.92, 2.44, 0.6, 0.48, 0.24, 0.44, 1.12, 0.16, 0.04, 0.96, 0.12, 1.92, 0.64, 1.4, 1.08, 0.08, 0.04, 0.88, 2.88, 2.92, 2.72, 3.4, 1.96, 0.08, 0.28, 2.72, 2.6, 1.84, 1.28, 1.44, 1.56, 1.04, 0.96, 0.08, 1.2, 0.08, 1.04, 0.4, 4.0, 0.4, 1.16, 2.28, 1.36, 0.8, 0.56, 1.96, 0.2, 1.04, 0.32, 2.56, 1.48, 0.6, 0.16, 1.96, 0.56, 1.32, 3.6, 2.48, 1.68, 3.44, 2.48, 2.96, 0.8, 1.44, 0.36, 0.76, 1.2, 0.76, 2.0, 1.52, 1.56, 0.24, 0.52, 1.04, 0.56, 2.04, 0.8, 0.68, 1.32, 3.4, 1.12, 0.16, 2.0, 4.0, 0.96, 0.28, 2.28, 0.64, 2.0, 2.16, 1.2, 0.16, 0.08, 0.88, 2.04, 1.44, 1.96, 0.52, 1.32, 0.44, 0.4, 2.56, 1.0, 2.28, 0.64, 0.32, 2.0, 3.08, 1.72, 1.2, 1.28, 2.52, 0.16, 1.0, 2.36, 0.76, 0.52, 0.48, 3.28, 0.84, 2.8, 0.84, 0.24, 1.04, 0.68, 1.92, 1.84, 0.96, 0.24, 0.48, 1.28, 0.72, 0.84, 0.12, 0.36, 0.08, 0.48, 0.04, 0.4, 1.4, 0.96, 2.08, 2.0, 1.44, 1.52, 4.6, 0.52, 1.84, 0.32, 2.24, 0.12, 0.6, 2.56, 1.56, 2.08, 1.56, 1.8, 1.68, 1.56, 5.04, 0.16, 0.36, 1.64, 3.12, 1.24, 1.28, 1.08, 1.84, 1.04, 2.68, 2.56, 2.2, 0.36, 1.04, 1.0, 1.0, 1.2, 1.16, 2.56, 1.36, 1.68, 2.24, 0.6, 1.8, 0.44, 0.2, 1.12, 2.04, 3.92, 1.8, 0.72, 3.16, 0.88, 2.68, 0.4, 0.68, 0.68, 4.32, 0.32, 3.36, 1.28, 3.64, 1.32, 3.2, 5.96, 1.92, 1.6, 1.96, 2.4, 0.24, 1.2, 1.24, 1.44, 1.6, 2.52, 3.2, 1.0, 2.36, 1.6, 2.6, 0.52, 1.28, 3.48, 2.48, 1.8, 3.36, 0.12, 2.72, 1.48, 2.4, 2.96, 0.24, 1.68, 1.84, 0.12, 1.16, 1.36, 0.12, 3.04, 4.44, 0.92, 0.76, 0.96, 2.88, 2.48, 1.4, 0.84, 2.88, 0.24, 1.52, 0.76, 2.96, 3.32, 2.16, 0.4, 1.76, 0.28, 4.08, 0.48, 1.24, 5.04, 4.16, 1.52, 0.16, 2.76, 2.84, 2.12, 0.92, 0.08, 0.8, 2.36, 2.56, 1.72, 0.16, 1.16, 0.72, 5.6, 0.88, 1.32, 0.76, 1.36, 2.04, 1.16, 1.44, 1.6, 1.28, 0.28, 0.92, 0.8, 0.92, 2.2, 4.52, 0.76, 2.88, 3.48, 1.48, 1.28, 0.36, 1.2, 1.36, 0.84, 1.12, 0.04, 2.92, 0.52, 1.6, 2.56, 1.36, 0.04, 0.4, 4.4, 2.88, 2.04, 3.16, 1.16, 1.32, 3.64, 0.52, 3.68, 0.84, 3.04, 0.28, 0.08, 1.8, 1.96, 3.12, 1.6, 0.36, 1.56, 0.48, 2.96, 2.56, 0.6, 0.12, 1.48, 1.68, 3.32, 2.28, 1.2, 0.88, 3.36, 1.24, 2.32, 1.8, 0.6, 2.24, 2.32, 1.04, 5.52, 0.4, 1.76, 1.04, 1.56, 1.2, 1.44, 1.2, 0.56, 0.12, 2.28, 0.8, 2.12, 2.0, 0.92, 0.4, 1.68, 1.24, 2.0, 0.28, 0.44, 0.08, 2.12, 0.48, 1.48, 0.16, 3.32, 0.28, 0.04, 0.08, 0.36, 0.2, 1.36, 0.28, 2.2, 1.44, 1.04, 0.12, 1.2, 1.88, 0.04, 2.84, 2.08, 0.04, 0.32, 0.28, 1.0, 0.64, 1.28, 2.08, 0.72, 0.04, 3.56, 1.4, 0.68, 2.64, 2.96, 0.44, 0.84, 1.72, 1.2, 2.2, 0.56, 0.76, 1.84, 0.68, 0.04, 0.28, 2.32, 0.08, 0.72, 1.04, 0.6, 4.16, 2.28, 3.12, 1.28, 4.0, 1.36, 2.08, 0.72, 3.52, 1.08, 0.32, 2.0, 1.76, 0.56, 4.32, 1.04, 0.72, 1.24, 3.4, 0.96, 2.12, 0.8, 1.04, 1.72, 1.56, 1.28, 5.92, 0.12, 0.48, 0.08, 2.96, 1.2, 0.68, 1.64, 1.88, 3.44, 1.0, 2.0, 0.48, 1.24, 2.08, 1.08, 2.8, 0.68, 2.76, 0.88, 1.8, 0.36, 1.36, 5.56, 1.88, 1.0, 0.16, 1.36, 1.56, 1.08, 0.32, 0.64, 0.96, 2.12, 0.4, 1.8, 2.72, 0.84, 3.48, 1.28, 0.88, 1.28, 0.84, 0.2, 1.88, 0.64, 4.48, 2.96, 4.0, 2.08, 4.76, 4.2, 2.96, 1.92, 1.16, 1.12, 2.4, 1.0, 1.0, 0.48, 1.44, 0.44, 1.96, 8.76, 2.28, 0.04, 0.12, 3.6, 1.2, 2.48, 1.2, 1.48, 1.52, 0.12, 1.24, 1.32, 0.2, 0.32, 3.68, 0.92, 0.8, 3.4, 2.16, 3.76, 1.56, 0.16, 5.2, 1.36, 0.8, 1.32, 0.16, 1.28, 1.36, 0.36, 2.4, 1.92, 3.48, 0.12, 2.76, 2.72, 0.44, 1.88, 3.44, 0.24, 1.52, 0.96, 1.56, 1.6, 2.72, 1.24, 0.4, 1.6, 0.8, 0.6, 0.92, 2.76, 6.88, 1.28, 2.4, 0.16, 1.68, 0.2, 1.72, 1.36, 0.92, 0.76, 1.8, 4.04, 3.32, 1.04, 1.48, 2.36, 1.04, 0.96, 0.8, 2.2, 0.48, 1.76, 2.64, 1.6, 4.28, 0.92, 0.84, 4.16, 4.68, 2.32, 1.44, 0.76, 3.28, 1.72, 2.8, 3.6, 3.84, 4.08, 1.48, 2.88, 0.56, 2.96, 2.76, 0.72, 4.04, 1.36, 1.12, 0.92, 0.84, 1.16, 0.6, 0.88, 0.44, 2.4, 0.44, 0.12, 2.6, 2.04, 5.72, 2.16, 1.04, 1.0, 2.88, 2.16, 0.12, 1.2, 4.88, 1.52, 1.44, 2.52, 0.24, 0.04, 1.96, 3.28, 0.12, 1.16, 1.28, 0.28, 0.72, 0.08, 1.32, 0.12, 0.2, 2.4, 0.2, 0.28, 0.48, 0.16, 0.32, 0.92, 2.2, 0.24, 0.92, 1.2, 0.96, 0.84, 2.12, 4.24, 2.68, 1.68, 1.68, 0.76, 2.36, 0.8, 1.04, 1.64, 3.48, 0.44, 2.08, 5.04, 1.44, 0.16, 0.68, 0.2, 2.72, 0.28, 1.8, 1.12, 4.48, 0.72, 1.56, 1.12, 0.48, 1.44, 2.04, 0.96, 0.28, 0.84, 3.12, 0.16, 1.96, 0.16, 0.8, 4.72, 0.36, 3.32, 2.04, 1.56, 0.76, 0.96, 1.12, 0.44, 1.04, 0.36, 1.0, 1.24, 0.56, 2.36, 0.12, 2.44, 4.08, 0.56, 0.88, 1.48, 0.72, 0.32, 1.88, 1.16, 0.04, 3.52, 4.0, 5.04, 1.76, 0.64, 2.28, 1.0, 5.76, 1.56, 6.44, 0.68, 0.68, 0.56, 2.12, 1.52, 1.2, 0.44, 2.68, 0.12, 2.28, 0.28, 0.12, 0.6, 1.36, 2.32, 0.56, 1.56, 0.32, 0.16, 0.92, 2.88, 0.8, 1.36, 0.04, 1.48, 0.08, 1.2, 0.12, 4.6, 0.08, 2.0, 1.8, 0.08, 3.68, 3.24, 2.24, 1.64, 0.64, 1.52, 0.2, 2.52, 1.16, 0.96, 0.8, 0.72, 0.72, 0.8, 1.76, 1.72, 0.52, 1.48, 0.24, 2.08, 0.96, 2.08, 0.84, 0.8, 3.32, 2.04, 1.76, 2.2, 2.76, 0.76, 0.76, 0.88, 1.68, 0.88, 2.2, 0.24, 2.72, 0.6, 0.04, 1.64, 0.44, 1.24, 0.84, 1.32, 1.32, 3.24, 1.4, 1.24, 2.96, 1.68, 1.48, 0.64, 2.76, 0.2, 3.0, 1.0, 2.16, 0.52, 1.52, 0.28, 1.72, 0.56, 0.16, 0.6, 1.2, 4.72, 2.68, 0.28, 2.28, 1.52, 0.8, 0.88, 2.88, 0.68, 0.16, 1.0, 1.8, 1.04, 1.16, 2.64, 2.6, 0.12, 1.72, 0.64, 1.64, 2.8, 0.16, 3.12, 2.24, 2.2, 2.24, 2.08, 2.4, 2.2, 0.76, 0.96, 0.8, 0.12, 0.16, 0.48, 1.96, 0.6, 1.24, 0.16, 1.4, 0.2, 0.2, 1.76, 0.32, 0.88, 0.84, 1.32, 0.72, 0.64, 2.64, 5.76, 1.12, 2.52, 0.44, 0.76, 1.76, 0.8, 0.56, 1.28, 0.76, 1.84, 0.6, 1.2, 0.56, 1.04, 0.16, 4.24, 2.44, 0.36, 0.44, 1.52, 1.4, 1.8, 2.28, 3.08, 0.52, 0.32, 2.32, 0.68, 2.56, 1.8, 2.4, 4.8, 2.72, 2.6, 2.44, 4.4, 4.12, 0.84, 3.0, 0.4, 1.76, 2.24, 1.6, 1.4, 0.8, 2.2, 0.44, 1.68, 1.0, 0.68, 0.04, 0.24, 0.28, 2.92, 1.2, 0.48, 1.16, 2.12, 1.48, 0.24, 1.88, 1.36, 4.04, 1.96, 0.72, 0.12, 2.64, 0.64, 2.2, 0.04, 5.56, 0.16, 1.52, 2.6, 1.72, 1.16, 0.32, 1.88, 2.56, 2.8, 1.08, 1.16, 1.64, 2.36, 1.64, 0.6, 0.28, 0.72, 2.6, 0.96, 1.52, 0.4, 0.52, 0.08, 1.56, 1.88, 3.12, 0.36, 3.32, 0.24, 0.32, 1.6, 2.88, 0.64, 2.36, 0.16, 1.92, 2.04, 1.04, 1.48, 0.16, 1.48, 2.04, 1.72, 2.28, 3.0, 1.0, 1.68, 0.24, 2.08, 2.0, 0.84, 0.6, 1.72, 1.08, 2.24, 4.68, 1.76, 0.32, 0.72, 1.24, 0.64, 1.2, 0.88, 1.0, 0.84, 0.88, 4.36, 0.64, 3.2, 0.84, 1.12, 0.16, 2.8, 2.64, 1.4, 0.04, 3.0, 2.32, 1.32, 1.24, 1.56, 0.68, 2.52, 0.8, 1.64, 3.24, 1.72, 0.44, 2.24, 2.0, 1.28, 0.64, 1.4, 0.68, 1.16, 0.32, 3.04, 1.44, 0.48, 1.8, 2.08, 2.72, 1.16, 2.72, 0.6, 3.2, 1.84, 0.52, 3.64, 1.72, 1.4, 0.92, 1.64, 2.64, 0.12, 1.4, 1.88, 1.84, 2.32, 1.0, 1.52, 0.96, 0.44, 1.16, 3.8, 0.56, 1.28, 3.08, 2.88, 0.48, 1.64, 0.64, 2.68, 1.16, 0.64, 2.16, 2.08, 1.52, 1.52, 2.2, 4.4, 1.24, 0.04, 1.92, 0.52, 0.52, 0.84, 1.64, 2.24, 0.6, 4.08, 1.16, 0.92, 0.52, 0.96, 1.8, 0.44, 1.72, 1.44, 0.36, 4.12, 0.88, 0.48, 1.88, 3.68, 1.48, 1.84, 0.96, 1.8, 1.88, 0.84, 1.0, 0.76, 1.4, 0.12, 1.12, 0.92, 2.56, 0.92, 0.88, 2.6, 0.12, 1.92, 2.12, 0.36, 1.76, 1.84, 0.28, 0.52, 2.68, 2.76, 0.92, 1.84, 0.48, 1.96, 0.6, 2.96, 0.84, 2.68, 1.4, 1.76, 1.0, 3.88, 0.6, 0.2, 2.0, 3.08, 1.52, 1.24, 4.56, 0.52, 0.16, 1.72, 2.24, 0.72, 1.44, 0.32, 2.2, 1.12, 0.16, 1.2, 2.48, 2.24, 1.6, 1.32, 1.56, 1.84, 1.92, 1.32, 2.28, 1.48, 0.44, 0.8, 0.96, 1.04, 0.04, 4.44, 2.96, 0.6, 0.92, 1.64, 1.84, 2.96, 1.0, 3.28, 0.72, 1.68, 0.72, 2.48, 0.12, 0.64, 0.68, 2.64, 0.12, 1.28, 2.4, 0.92, 2.8, 0.4, 1.88, 0.8, 2.44, 1.2, 0.52, 1.2, 2.56, 2.64, 1.64, 1.84, 1.16, 1.72, 0.08, 3.84, 1.6, 1.0, 0.12, 0.44, 0.56, 2.32, 0.64, 0.64, 1.28, 0.6, 2.08, 2.72, 2.92, 0.8, 1.76, 0.8, 0.88, 0.96, 0.16, 0.8, 0.92, 1.48, 1.76, 1.72, 1.68, 0.96, 2.44, 0.88, 2.04, 1.16, 2.6, 1.2, 1.12, 1.76, 1.08, 1.48, 1.32, 1.12, 1.48, 0.08, 2.32, 1.92, 0.16, 1.0, 3.12, 0.32, 1.64, 0.68, 0.32, 1.32, 1.12, 0.32, 1.04, 1.96, 0.44, 2.76, 0.24, 0.96, 1.68, 0.96, 0.8, 2.12, 0.08, 0.08, 3.0, 0.12, 1.04, 0.48, 0.8, 2.6, 2.88, 1.08, 1.12, 0.76, 1.36, 0.96, 2.0, 0.04, 1.88, 1.4, 0.48, 1.04, 1.08, 0.84, 1.16, 0.24, 2.44, 1.88, 0.68, 3.28, 2.36, 1.4, 0.12, 0.76, 1.16, 0.48, 1.88, 5.0, 2.12, 0.76, 0.76, 2.96, 0.56, 3.0, 0.36, 3.08, 0.24, 1.4, 1.16, 0.16, 0.32, 1.8, 0.04, 2.16, 2.64, 0.56, 1.8, 0.76, 1.92, 1.0, 0.36, 0.6, 0.68, 0.16, 0.52, 1.44, 1.0, 1.64, 2.48, 0.44, 0.2, 0.64, 0.8, 0.56, 2.8, 3.4, 0.08, 3.24, 0.88, 0.64, 2.52, 1.96, 1.32, 0.68, 1.2, 2.8, 0.2, 1.12, 0.04, 1.48, 1.32, 1.84, 3.04, 1.16, 2.4, 0.92, 1.64, 0.88, 3.72, 0.8, 3.72, 2.92, 3.12, 3.88, 2.52, 0.52, 0.6, 1.0, 3.12, 1.68, 5.24, 3.12, 0.24, 1.32, 0.4, 5.76, 2.96, 0.88, 5.08, 0.6, 2.12, 0.08, 0.84, 0.8, 2.36, 1.88, 0.36, 1.56, 1.12, 4.64, 0.44, 0.12, 6.88, 2.52, 2.28, 1.68, 2.84, 1.44, 0.4, 2.16, 4.88, 2.56, 0.88, 1.76, 1.04, 1.24, 2.04, 3.48, 1.4, 0.6, 0.8, 1.0, 5.96, 3.96, 0.92, 5.52, 1.88, 2.64, 4.08, 1.6, 1.88, 0.04, 1.76, 0.84, 0.6, 2.24, 1.48, 2.4, 1.24, 5.24, 2.36, 0.8, 2.8, 1.64, 1.32, 1.56, 2.52, 3.96, 0.76, 2.76, 1.24, 4.76, 4.04, 1.12, 0.68, 6.04, 0.8, 2.2, 2.92, 0.88, 2.2, 0.44, 2.44, 4.44, 0.92, 1.76, 0.2, 2.28, 1.32, 1.64, 0.08, 0.16, 2.76, 1.8, 1.68, 0.24, 4.64, 4.24, 1.2, 0.24, 1.24, 1.08, 0.96, 0.84, 0.52]
segments_tuned_length = [1.64, 4.0, 6.48, 0.96, 0.48, 1.04, 0.68, 0.56, 1.76, 2.4, 2.64, 4.28, 5.24, 2.4, 3.88, 2.24, 2.48, 1.28, 0.76, 0.96, 0.64, 0.12, 0.96, 2.0, 0.8, 1.36, 7.92, 1.56, 1.56, 0.76, 0.72, 0.52, 0.6, 4.24, 3.96, 0.72, 1.08, 3.2, 2.88, 0.08, 0.36, 1.76, 0.88, 1.04, 1.4, 0.48, 3.08, 0.36, 2.24, 1.68, 2.68, 1.2, 0.64, 0.68, 1.4, 0.12, 0.36, 3.4, 2.64, 2.48, 1.04, 0.96, 0.76, 2.96, 1.36, 3.48, 1.24, 2.52, 0.24, 2.84, 1.84, 1.12, 0.84, 2.68, 5.52, 0.48, 1.64, 3.2, 1.24, 2.44, 1.4, 0.6, 0.8, 1.16, 1.6, 2.6, 0.36, 1.16, 0.24, 1.16, 2.32, 3.92, 2.2, 0.4, 2.44, 1.36, 0.12, 1.16, 2.48, 0.44, 1.68, 4.68, 2.72, 1.28, 1.28, 5.24, 7.84, 0.6, 2.44, 0.36, 1.36, 0.44, 2.08, 2.52, 1.04, 0.96, 0.8, 2.8, 0.16, 2.88, 0.96, 2.76, 1.8, 0.92, 2.04, 2.76, 2.44, 3.56, 0.76, 0.72, 1.36, 1.48, 1.6, 0.2, 2.28, 1.84, 0.32, 0.76, 2.16, 1.72, 0.28, 1.24, 2.4, 0.8, 0.76, 1.52, 1.76, 2.24, 5.0, 5.0, 0.12, 2.44, 2.16, 0.2, 2.56, 2.92, 0.48, 0.12, 2.24, 1.84, 1.24, 2.92, 2.6, 1.2, 5.36, 0.44, 2.8, 0.32, 0.36, 1.8, 0.84, 1.44, 1.68, 1.92, 0.92, 0.32, 2.92, 3.12, 2.72, 1.32, 0.32, 0.48, 1.2, 0.2, 0.12, 1.28, 2.2, 0.12, 0.64, 1.48, 1.12, 0.04, 0.84, 2.92, 5.64, 3.4, 2.0, 0.32, 5.72, 1.84, 1.28, 1.44, 1.68, 1.12, 1.04, 0.4, 1.48, 0.08, 1.08, 0.56, 4.08, 0.56, 3.68, 1.96, 1.6, 1.96, 0.2, 1.12, 0.52, 2.6, 1.6, 0.56, 0.16, 1.96, 0.56, 1.32, 6.16, 5.24, 2.44, 2.96, 0.92, 2.96, 2.32, 1.16, 0.8, 2.0, 1.6, 1.52, 0.24, 2.76, 1.12, 0.56, 2.04, 0.92, 0.68, 1.32, 3.44, 1.16, 0.56, 2.08, 4.12, 0.96, 0.36, 2.4, 0.68, 2.04, 2.2, 1.48, 0.08, 0.92, 2.04, 1.56, 2.0, 0.6, 1.32, 0.44, 0.56, 2.72, 3.44, 0.88, 0.44, 2.0, 3.36, 1.76, 1.24, 1.4, 2.88, 0.2, 1.0, 2.36, 1.36, 0.48, 3.48, 0.84, 2.8, 0.96, 2.48, 1.92, 1.84, 1.0, 0.24, 0.48, 1.28, 1.0, 0.92, 0.72, 0.44, 0.24, 0.56, 0.44, 4.56, 2.0, 1.6, 1.56, 7.04, 0.56, 3.72, 0.16, 0.68, 4.32, 2.08, 6.8, 5.16, 0.32, 1.72, 3.28, 2.56, 1.08, 3.0, 2.92, 4.8, 1.44, 1.12, 1.0, 1.2, 1.36, 2.52, 1.64, 1.72, 2.24, 0.88, 2.08, 0.44, 0.32, 1.16, 2.12, 5.8, 1.32, 3.24, 1.12, 2.76, 0.36, 0.84, 0.72, 4.32, 0.32, 3.36, 1.48, 3.6, 1.36, 3.6, 6.52, 2.04, 1.8, 1.96, 2.44, 0.2, 1.36, 1.72, 3.12, 2.68, 3.2, 1.24, 4.24, 2.56, 0.52, 1.36, 3.56, 2.72, 1.84, 3.6, 2.88, 1.48, 2.44, 3.08, 0.56, 1.88, 2.56, 1.24, 1.56, 0.28, 3.28, 4.64, 0.92, 0.8, 1.08, 2.88, 2.56, 2.36, 3.04, 2.24, 0.84, 3.04, 3.52, 2.24, 2.44, 0.28, 4.4, 0.52, 1.52, 5.04, 4.24, 1.96, 0.2, 3.12, 2.84, 2.12, 1.08, 0.16, 0.96, 2.76, 4.36, 0.16, 2.36, 5.8, 0.92, 1.48, 1.12, 1.32, 2.12, 1.4, 3.12, 1.28, 0.28, 0.92, 2.04, 2.36, 4.52, 0.72, 2.92, 3.6, 1.76, 1.68, 1.32, 1.36, 2.08, 0.16, 3.04, 0.64, 1.76, 2.52, 1.68, 0.36, 4.6, 2.88, 2.2, 3.16, 1.2, 1.36, 4.36, 3.76, 0.88, 3.44, 0.32, 2.24, 1.96, 3.12, 2.2, 1.56, 0.6, 3.44, 0.68, 1.24, 1.48, 1.68, 3.4, 2.4, 2.28, 3.44, 1.36, 2.28, 1.8, 0.68, 5.68, 5.52, 0.48, 1.72, 1.16, 1.6, 1.16, 1.52, 1.36, 0.56, 0.44, 3.56, 2.16, 3.08, 2.24, 1.32, 2.12, 0.8, 0.56, 0.2, 2.16, 0.6, 1.56, 0.16, 3.76, 0.44, 0.68, 2.16, 1.36, 0.28, 2.24, 1.4, 1.12, 0.04, 1.12, 2.08, 2.92, 2.24, 0.08, 0.36, 0.32, 1.24, 0.64, 1.4, 2.36, 0.84, 0.12, 3.56, 1.4, 0.68, 2.64, 3.0, 0.44, 2.6, 1.72, 2.96, 0.8, 2.64, 0.84, 0.12, 0.4, 2.36, 0.28, 0.72, 1.04, 0.64, 4.12, 2.44, 3.16, 4.0, 1.4, 2.08, 0.92, 3.72, 1.12, 1.12, 2.24, 1.76, 5.12, 1.04, 1.0, 1.32, 3.52, 1.12, 2.12, 2.16, 0.04, 1.68, 1.76, 1.28, 6.16, 3.0, 0.24, 2.96, 1.4, 0.68, 1.64, 1.88, 4.56, 2.0, 0.48, 1.36, 2.16, 1.08, 2.8, 0.76, 2.8, 3.0, 0.36, 1.36, 5.56, 1.92, 1.12, 0.16, 1.44, 1.52, 0.08, 0.96, 0.32, 0.64, 1.08, 2.12, 5.04, 0.8, 4.8, 0.96, 1.28, 1.44, 2.64, 4.96, 3.12, 4.04, 2.16, 4.84, 4.2, 2.96, 2.2, 1.4, 3.68, 1.2, 1.08, 0.56, 1.44, 0.44, 2.08, 8.8, 2.52, 3.68, 1.2, 3.76, 1.56, 1.72, 0.08, 1.36, 1.32, 0.2, 0.4, 3.72, 1.04, 0.8, 3.48, 2.16, 3.76, 1.64, 0.12, 5.32, 1.64, 0.8, 1.32, 4.48, 0.36, 2.72, 1.92, 3.96, 2.88, 2.72, 0.44, 5.76, 1.56, 0.92, 1.68, 1.64, 2.76, 1.24, 0.6, 1.56, 0.8, 0.96, 0.92, 3.28, 6.88, 1.28, 2.44, 1.88, 0.24, 1.72, 1.4, 0.88, 0.8, 1.8, 4.04, 3.44, 2.68, 0.04, 2.24, 1.36, 1.08, 0.88, 2.2, 0.48, 1.76, 2.84, 1.6, 4.56, 0.96, 0.88, 4.24, 4.84, 2.32, 1.44, 0.92, 3.4, 4.6, 7.52, 5.88, 2.88, 0.64, 2.96, 2.76, 0.8, 4.08, 1.36, 1.12, 1.04, 0.88, 1.16, 0.68, 0.88, 0.44, 2.52, 0.64, 2.64, 2.16, 6.52, 2.2, 2.36, 4.04, 2.16, 0.12, 8.0, 1.6, 2.96, 2.0, 3.36, 0.32, 1.16, 1.4, 0.36, 1.6, 0.16, 0.04, 1.44, 0.12, 0.4, 2.68, 0.64, 0.72, 0.32, 0.28, 1.04, 2.24, 0.24, 1.0, 1.4, 0.96, 0.96, 2.12, 4.32, 2.88, 1.8, 1.68, 1.96, 2.4, 0.8, 1.12, 1.8, 3.48, 0.52, 7.76, 1.52, 0.32, 0.84, 0.36, 3.56, 2.24, 1.12, 4.6, 2.32, 1.64, 1.52, 2.0, 1.0, 0.36, 0.92, 3.36, 0.16, 2.08, 0.24, 0.88, 5.32, 3.52, 2.08, 1.52, 0.76, 0.96, 1.12, 1.72, 0.6, 1.08, 1.24, 0.64, 3.16, 0.16, 2.52, 4.12, 1.72, 1.72, 0.88, 0.68, 1.88, 1.36, 3.56, 4.12, 5.04, 2.4, 0.64, 2.28, 1.16, 7.96, 6.44, 0.68, 0.68, 0.6, 2.12, 1.52, 1.24, 0.56, 2.76, 0.12, 2.68, 0.16, 0.64, 1.36, 2.32, 0.6, 2.4, 0.56, 0.2, 0.88, 3.0, 1.0, 1.44, 1.56, 0.2, 1.2, 0.08, 4.76, 0.32, 2.0, 1.92, 3.84, 7.32, 2.44, 1.72, 0.72, 1.72, 3.96, 1.8, 2.08, 0.92, 3.6, 0.68, 1.56, 0.24, 2.08, 0.96, 3.84, 5.36, 1.8, 5.0, 1.76, 0.96, 1.64, 1.04, 2.2, 0.24, 2.72, 0.76, 0.12, 1.76, 1.72, 1.04, 1.48, 1.32, 3.24, 1.48, 0.04, 1.16, 3.04, 1.72, 1.6, 0.64, 2.76, 0.24, 3.32, 1.0, 2.4, 0.48, 1.56, 2.32, 0.56, 0.16, 0.6, 1.2, 4.72, 2.72, 0.32, 2.28, 1.6, 1.0, 0.96, 3.08, 0.88, 2.84, 1.04, 1.44, 2.84, 2.8, 1.92, 2.4, 2.88, 0.2, 3.2, 2.4, 2.4, 2.92, 2.12, 2.4, 2.24, 0.72, 0.96, 1.48, 0.28, 0.16, 0.52, 2.48, 0.6, 1.32, 0.2, 2.36, 0.72, 2.12, 0.36, 0.84, 0.84, 1.32, 0.8, 0.64, 2.64, 6.0, 1.12, 2.68, 0.44, 1.08, 2.6, 0.68, 1.4, 0.92, 2.0, 0.6, 1.28, 0.56, 1.04, 0.16, 4.24, 2.84, 0.56, 1.56, 3.28, 2.6, 3.08, 0.52, 3.08, 0.68, 2.64, 1.96, 7.16, 0.12, 4.72, 2.88, 2.64, 2.52, 4.44, 4.08, 0.88, 3.0, 1.12, 1.92, 2.32, 1.6, 1.4, 0.8, 2.2, 0.44, 2.88, 0.84, 0.4, 0.32, 0.36, 3.08, 1.72, 1.32, 2.2, 1.64, 0.28, 1.88, 1.52, 4.08, 2.2, 0.96, 0.12, 2.64, 1.12, 2.48, 5.56, 0.2, 1.56, 2.6, 1.72, 1.24, 0.52, 1.84, 2.6, 2.88, 1.08, 1.28, 1.64, 2.4, 2.28, 1.04, 3.16, 0.96, 1.96, 0.92, 0.12, 1.6, 2.04, 3.12, 0.4, 3.32, 0.68, 1.68, 2.88, 0.76, 2.36, 0.24, 4.12, 1.0, 1.52, 1.72, 2.0, 1.76, 2.32, 3.08, 0.96, 1.84, 0.24, 2.08, 2.04, 0.8, 0.6, 1.72, 1.16, 2.28, 4.72, 2.08, 0.72, 1.36, 0.64, 1.32, 0.88, 1.0, 0.92, 0.92, 4.44, 0.72, 3.2, 1.0, 1.4, 2.8, 2.64, 1.52, 3.04, 2.56, 1.56, 1.2, 1.72, 0.84, 3.36, 1.64, 3.32, 1.84, 0.48, 2.24, 2.08, 3.48, 0.72, 1.48, 0.68, 1.16, 3.56, 1.72, 0.48, 1.92, 2.08, 2.84, 1.24, 2.8, 0.6, 3.52, 1.88, 0.64, 5.48, 1.4, 0.92, 1.68, 2.8, 1.56, 3.84, 2.32, 1.28, 2.52, 0.48, 1.2, 3.8, 0.64, 1.36, 3.2, 2.84, 0.68, 1.8, 0.76, 3.92, 0.84, 2.24, 2.16, 1.76, 1.52, 2.32, 4.76, 2.24, 0.16, 1.96, 0.8, 3.12, 2.2, 0.88, 4.12, 1.16, 0.92, 1.32, 2.88, 2.2, 1.44, 0.36, 5.2, 6.28, 1.72, 1.84, 0.96, 1.92, 1.88, 3.72, 2.08, 1.36, 2.24, 1.4, 3.6, 0.92, 0.88, 2.76, 2.0, 2.12, 0.36, 1.76, 1.84, 0.8, 0.92, 2.84, 2.8, 1.44, 1.84, 2.52, 1.08, 2.96, 0.84, 2.8, 1.4, 1.76, 1.12, 3.84, 0.76, 2.32, 3.2, 1.56, 5.92, 0.52, 2.0, 2.24, 2.2, 0.4, 2.36, 1.12, 0.16, 1.2, 2.48, 2.24, 1.6, 1.32, 3.48, 2.36, 1.36, 2.36, 1.48, 0.44, 0.8, 0.96, 1.08, 4.52, 3.56, 1.36, 3.52, 3.0, 1.0, 4.2, 1.72, 1.24, 2.8, 1.36, 2.68, 0.16, 1.36, 2.36, 1.12, 3.36, 1.88, 0.88, 2.48, 1.2, 0.52, 1.24, 2.72, 2.76, 1.72, 1.88, 3.0, 0.32, 3.84, 1.64, 1.4, 0.2, 0.6, 2.96, 0.6, 0.88, 2.08, 0.64, 2.16, 2.76, 2.88, 0.84, 2.12, 0.96, 1.0, 1.32, 0.76, 2.56, 1.76, 3.4, 0.96, 2.44, 1.0, 2.12, 1.24, 2.64, 1.36, 1.28, 1.8, 1.08, 1.6, 1.32, 1.16, 1.56, 0.16, 2.68, 1.92, 0.16, 1.08, 3.12, 0.32, 2.36, 1.8, 1.24, 0.32, 1.12, 2.44, 4.24, 1.68, 1.04, 0.76, 2.16, 0.48, 3.0, 0.12, 1.24, 1.52, 2.72, 3.16, 1.04, 1.44, 1.12, 1.6, 0.96, 2.52, 1.92, 2.0, 1.04, 1.12, 1.12, 1.2, 0.2, 2.44, 2.04, 0.84, 3.28, 2.36, 1.6, 0.12, 2.04, 0.48, 1.92, 5.0, 2.24, 0.72, 0.76, 2.96, 0.56, 3.0, 0.44, 3.32, 0.28, 1.36, 1.16, 0.52, 1.92, 0.2, 2.2, 2.72, 0.56, 2.0, 0.88, 0.04, 1.84, 1.0, 1.0, 0.72, 0.72, 1.68, 1.08, 1.72, 2.56, 0.44, 1.04, 0.8, 0.68, 2.8, 3.56, 0.04, 3.24, 1.48, 0.68, 2.56, 2.0, 1.32, 0.68, 1.2, 2.8, 0.28, 1.16, 1.72, 1.32, 1.92, 3.92, 1.28, 3.36, 1.64, 0.96, 3.84, 0.96, 3.08, 3.56, 3.88, 2.68, 0.52, 0.6, 1.2, 3.12, 7.0, 3.12, 0.48, 1.52, 6.4, 2.96, 0.88, 5.12, 2.76, 0.08, 0.84, 0.84, 2.32, 2.08, 0.36, 2.72, 5.32, 6.84, 2.6, 2.28, 1.76, 2.84, 1.44, 0.44, 2.16, 4.88, 2.64, 0.88, 1.76, 1.04, 1.28, 2.04, 3.52, 1.4, 0.6, 1.92, 5.96, 4.04, 1.04, 5.56, 1.88, 2.72, 4.12, 1.6, 1.96, 0.12, 1.92, 0.92, 0.8, 2.72, 1.64, 2.48, 1.24, 5.4, 2.36, 0.8, 2.8, 1.76, 1.68, 1.56, 2.6, 3.96, 1.24, 4.16, 4.8, 4.36, 1.12, 0.76, 7.0, 2.2, 4.88, 1.28, 2.4, 0.48, 2.44, 4.48, 6.88, 1.84, 0.12, 3.0, 1.92, 2.12, 4.64, 4.32, 1.28, 0.2, 1.4, 1.24, 0.96, 1.12, 0.56]
segments_gold_length = [2.48, 2.88, 6.92, 5.52, 0.92, 2.16, 6.92, 2.44, 1.4, 1.32, 2.0, 1.76, 1.84, 2.12, 2.12, 1.12, 1.88, 1.08, 1.56, 2.2, 1.12, 7.8, 1.08, 4.12, 4.04, 0.16, 1.12, 2.84, 2.72, 0.16, 1.36, 0.48, 0.64, 1.08, 3.8, 1.28, 2.48, 1.44, 2.28, 1.28, 1.16, 1.16, 1.72, 3.12, 2.64, 2.6, 3.16, 7.56, 4.08, 6.08, 1.84, 4.44, 4.36, 1.04, 2.24, 2.24, 1.36, 0.56, 0.64, 2.6, 2.28, 1.12, 2.32, 1.24, 0.6, 2.68, 1.6, 5.56, 3.24, 2.6, 0.84, 3.2, 4.76, 5.16, 7.0, 3.12, 2.96, 1.12, 2.32, 6.04, 1.32, 0.44, 0.76, 2.44, 2.24, 1.52, 8.72, 2.8, 1.6, 5.2, 2.92, 3.96, 3.92, 1.16, 2.12, 1.8, 0.68, 1.92, 0.28, 1.84, 1.4, 2.2, 1.48, 1.36, 1.08, 2.96, 2.04, 1.24, 1.64, 0.16, 0.2, 1.04, 0.44, 5.24, 5.32, 4.52, 1.76, 1.76, 1.12, 0.4, 3.08, 2.72, 2.92, 0.76, 1.44, 9.72, 3.44, 1.88, 8.44, 0.44, 3.44, 1.72, 0.48, 0.8, 1.04, 3.28, 2.04, 2.72, 0.76, 1.12, 1.28, 4.88, 3.24, 2.64, 2.96, 3.84, 8.6, 2.28, 1.2, 2.04, 1.24, 1.52, 0.68, 4.52, 3.44, 1.88, 4.68, 1.88, 5.08, 1.4, 2.76, 0.44, 4.2, 0.64, 4.08, 4.16, 2.6, 1.16, 0.64, 2.64, 2.48, 1.88, 3.4, 0.96, 1.04, 3.36, 1.04, 1.68, 1.28, 0.96, 2.76, 0.76, 2.0, 0.36, 0.92, 7.0, 0.8, 0.48, 5.32, 3.12, 5.0, 5.2, 2.96, 1.24, 1.16, 1.8, 1.16, 0.6, 0.56, 6.96, 2.96, 2.96, 2.12, 6.44, 4.24, 4.76, 1.96, 3.4, 1.44, 4.96, 1.48, 2.08, 6.44, 7.12, 2.48, 4.28, 4.6, 4.44, 3.24, 7.8, 4.4, 3.2, 8.56, 8.44, 4.4, 3.96, 1.44, 2.24, 4.04, 1.56, 2.92, 2.08, 2.56, 3.92, 3.16, 1.28, 2.28, 3.28, 6.36, 6.0, 2.04, 6.32, 4.08, 2.24, 8.96, 3.2, 2.44, 5.0, 7.16, 4.88, 1.16, 3.24, 2.16, 2.64, 4.64, 4.8, 4.12, 2.68, 4.84, 5.12, 3.72, 6.92, 3.32, 1.52, 5.12, 6.2, 5.84, 5.28, 4.96, 3.64, 3.72, 2.6, 5.72, 2.84, 1.76, 0.84, 6.56, 5.36, 0.96, 4.16, 0.84, 3.16, 6.0, 3.76, 2.48, 3.36, 4.52, 2.64, 1.12, 0.24, 1.36, 1.6, 2.44, 1.68, 2.56, 1.0, 1.4, 2.6, 3.72, 0.56, 2.56, 0.4, 0.28, 0.56, 0.52, 2.6, 0.48, 0.56, 1.08, 0.2, 3.4, 0.88, 2.36, 2.8, 0.2, 2.48, 1.72, 0.84, 0.12, 2.12, 2.32, 2.2, 1.2, 3.16, 2.52, 1.08, 2.68, 2.04, 1.0, 2.4, 0.56, 7.64, 1.0, 0.56, 1.72, 3.0, 0.16, 3.68, 3.36, 1.04, 1.28, 5.76, 5.76, 6.52, 3.52, 3.16, 3.2, 5.4, 1.72, 5.6, 4.32, 1.92, 2.52, 4.04, 0.8, 0.72, 1.6, 3.64, 2.12, 5.68, 2.36, 5.16, 2.0, 1.36, 1.44, 3.2, 5.12, 2.4, 0.64, 2.56, 5.96, 6.0, 6.48, 3.48, 3.24, 1.32, 2.44, 3.4, 3.6, 3.68, 6.64, 6.2, 4.52, 2.68, 2.64, 0.76, 1.04, 1.08, 3.16, 2.32, 2.48, 0.6, 4.88, 3.08, 1.52, 3.12, 0.96, 4.2, 5.08, 4.28, 1.76, 2.04, 1.76, 1.36, 0.96, 6.24, 4.76, 1.44, 5.4, 2.0, 1.84, 3.12, 0.72, 9.52, 2.32, 3.2, 6.68, 0.8, 1.48, 5.24, 0.88, 4.28, 4.56, 5.96, 5.76, 2.72, 4.44, 1.64, 7.28, 1.24, 3.16, 5.68, 1.8, 8.68, 1.48, 6.08, 1.96, 0.96, 1.84, 7.08, 0.76, 2.96, 7.44, 1.76, 1.96, 4.24, 8.28, 1.56, 1.08, 2.28, 1.56, 0.92, 0.6, 0.2, 1.48, 4.28, 2.56, 2.16, 0.2, 0.88, 2.92, 4.8, 2.44, 1.84, 4.72, 2.12, 2.6, 1.0, 0.2, 5.84, 0.16, 3.2, 6.0, 2.56, 3.76, 6.0, 2.92, 3.8, 3.12, 2.68, 0.68, 2.24, 2.12, 2.88, 4.72, 2.2, 2.08, 4.2, 3.0, 3.4, 2.32, 4.0, 1.56, 2.44, 3.12, 3.64, 3.8, 3.52, 4.84, 0.56, 5.76, 1.16, 6.76, 2.36, 3.52, 0.12, 2.96, 1.28, 0.64, 1.52, 0.24, 1.84, 2.2, 0.64, 1.24, 0.56, 2.84, 1.52, 6.04, 2.24, 1.4, 1.6, 0.28, 1.28, 0.08, 0.52, 5.6, 3.08, 2.96, 6.12, 0.56, 5.48, 0.8, 3.24, 4.2, 1.56, 2.8, 2.56, 1.76, 1.16, 2.96, 4.76, 2.64, 1.56, 1.28, 1.36, 1.6, 0.24, 2.0, 4.12, 1.96, 3.08, 0.52, 0.84, 0.72, 1.56, 1.0, 1.0, 3.56, 1.08, 1.32, 2.04, 1.68, 1.72, 2.2, 1.4, 1.76, 3.28, 2.12, 4.76, 2.44, 1.48, 0.92, 1.44, 1.68, 0.8, 2.48, 1.72, 1.48, 1.72, 1.4, 1.68, 3.6, 1.04, 0.52, 2.04, 4.32, 0.8, 2.72, 2.24, 2.2, 2.76, 3.36, 0.4, 2.72, 2.76, 0.44, 4.16, 1.56, 1.2, 1.76, 0.72, 3.0, 2.64, 2.12, 3.16, 3.64, 3.56, 2.32, 2.16, 2.4, 1.04, 4.0, 1.44, 1.4, 2.16, 3.28, 2.16, 1.12, 4.24, 0.84, 9.52, 2.84, 2.0, 2.44, 6.0, 6.44, 0.68, 7.52, 3.88, 5.44, 2.52, 2.6, 6.84, 2.04, 1.92, 1.48, 2.48, 1.68, 0.84, 3.04, 3.44, 11.0, 3.12, 1.6, 2.0, 5.08, 0.12, 3.64, 1.68, 0.6, 8.08, 6.76, 4.92, 2.16, 2.72, 2.48, 0.8, 3.16, 3.96, 5.16, 1.92, 3.12, 1.68, 0.8, 3.44, 1.44, 1.24, 4.92, 3.84, 0.92, 1.56, 3.6, 1.68, 1.88, 4.52, 4.36, 2.88, 2.04, 2.4, 1.36, 3.04, 2.88, 0.52, 4.64, 2.76, 3.84, 3.8, 3.36, 1.72, 8.12, 6.04, 2.36, 4.76, 2.16, 2.24, 3.52, 5.4, 4.04, 4.48, 3.56, 1.84, 6.8, 2.44, 1.76, 4.8, 3.12, 2.56, 6.24, 0.96, 0.92, 2.0, 3.32, 0.64, 2.44, 5.12, 1.64, 0.88, 4.32, 4.32, 3.56, 4.12, 2.04, 0.68, 0.76, 0.64, 3.92, 4.68, 3.08, 4.36, 2.84, 0.72, 0.72, 1.68, 2.28, 1.28, 1.12, 0.92, 0.88, 0.76, 0.72, 3.8, 2.68, 2.44, 2.96, 2.36, 2.32, 1.88, 1.12, 1.88, 1.24, 1.28, 0.76, 1.96, 1.96, 1.12, 0.48, 1.28, 1.56, 1.72, 0.96, 2.76, 0.88, 3.12, 1.32, 3.44, 1.04, 2.56, 0.68, 1.16, 1.08, 3.32, 1.32, 3.88, 3.76, 5.64, 2.12, 4.12, 4.72, 6.36, 7.56, 1.36, 2.88, 2.36, 3.8, 4.48, 1.4, 1.6, 0.92, 3.28, 2.36, 0.44, 2.84, 2.84, 1.2, 1.92, 1.96, 0.4, 1.32, 1.96, 4.68, 4.72, 1.44, 1.28, 2.56, 1.04, 0.4, 1.68, 1.8, 1.16, 2.72, 4.32, 3.92, 0.92, 2.68, 0.4, 0.96, 1.16, 1.68, 4.16, 0.48, 2.56, 1.04, 3.88, 3.0, 6.2, 2.36, 3.6, 2.12, 2.52, 2.8, 1.48, 4.6, 4.4, 1.96, 0.72, 4.4, 1.68, 2.2, 3.24, 2.24, 7.52, 0.76, 2.88, 1.12, 0.2, 2.76, 1.04, 7.6, 6.32, 6.52, 3.24, 3.16, 2.52, 6.2, 2.28, 4.68, 2.44, 6.68, 1.92, 4.44, 1.36, 2.72, 1.44, 2.56, 3.92, 2.6, 1.6, 6.88, 1.36, 2.56, 1.92, 2.68, 1.64, 1.52, 2.52, 0.88, 1.72, 3.44, 5.28, 4.48, 2.0, 2.56, 0.88, 2.12, 0.64, 3.24, 2.6, 2.8, 1.92, 3.84, 1.92, 6.68, 3.84, 1.6, 1.56, 0.52, 1.0, 1.84, 2.0, 2.56, 4.32, 1.68, 2.92, 2.28, 3.08, 2.76, 4.8, 1.92, 3.2, 3.56, 2.72, 2.52, 3.8, 1.76, 3.88, 2.84, 3.52, 3.88, 1.12, 2.24, 3.92, 1.76, 4.04, 3.28, 2.2, 0.68, 1.92, 2.52, 2.36, 3.04, 1.88, 1.28, 1.44, 4.04, 3.24, 2.28, 4.24, 1.64, 4.4, 2.48, 3.12, 2.6, 3.16, 4.56, 4.68, 2.24, 2.48, 1.84, 2.24, 1.04, 1.0, 4.0, 0.8, 6.56, 2.0, 1.76, 0.6, 1.92, 0.8, 3.0, 2.0, 1.48, 0.36, 1.2, 0.32, 0.44]

title = f"phrase segment length distribution" 
bins = 50
alpha = 0.5
max_value = 175
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.hist(segments_default_length, bins=bins, alpha=alpha, label="Default predicted segments")
plt.hist(segments_tuned_length, bins=bins, alpha=alpha, label="Tuned predicted segments")
plt.hist(segments_gold_length, bins=bins, alpha=alpha, label="Gold segments")
plt.subplots_adjust(bottom=0.1, top=0.98) 
plt.legend(loc="upper right", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlabel("Length in seconds", fontsize=18)
plt.ylabel("Number of segments", fontsize=18)
plt.ylim(0, max_value)

# plt.show()
plt.savefig("/Users/zifanjiang/Downloads/segments_distribution.pdf", format="pdf", bbox_inches="tight")