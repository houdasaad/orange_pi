import os
import numpy as np
import cv2
import torch
import tensorflow as tf
import time
import csv
from multiprocessing import Process, Queue
import psutil

# Configuración para usar solo CPU y suprimir warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # usar solo CPU para TensorFlow Lite
tf.get_logger().setLevel('ERROR')

from mcunet.utils.det_helper import MergeNMS, Yolo3Output
from mcunet.model_zoo import download_tflite

def get_model_info(tflite_path):
    interpreter = tf.lite.Interpreter(tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_dtype = input_details[0]['dtype']
    print(f"Tipo de datos de entrada: {input_dtype}")
    precision = str(input_dtype).split('.')[-1]

    model_size = os.path.getsize(tflite_path) / (1024 * 1024)  # en MB

    # Contar el número de tensores
    num_tensors = len(interpreter.get_tensor_details())

    return precision, model_size, num_tensors

def build_det_helper():
    nms = MergeNMS.build_from_config({
        "nms_name": "merge",
        "nms_valid_thres": 0.01,
        "nms_thres": 0.45,
        "nms_topk": 400,
        "post_nms": 100,
        "pad_val": -1,
    })
    output_configs = [
        {"num_class": 1, "anchors": [116, 90, 156, 198, 373, 326], "stride": 32, "alloc_size": [128, 128]},
        {"num_class": 1, "anchors": [30, 61, 62, 45, 59, 119], "stride": 16, "alloc_size": None},
        {"num_class": 1, "anchors": [10, 13, 16, 30, 33, 23], "stride": 8, "alloc_size": None},
    ]
    outputs = [
        Yolo3Output(**cfg).eval() for cfg in output_configs
    ]
    return nms, outputs

def preprocess_image(frame, resolution):
    resized = cv2.resize(frame, (resolution[1], resolution[0]))
    image_np = np.expand_dims(resized, axis=0)
    image_np = (image_np / 255) * 2 - 1  # Normalización según el modelo
    return image_np.astype('float32')

def eval_image(image, interpreter, input_details, output_details, output_layers, nms_layer):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    outputs = [torch.from_numpy(d).permute(0, 3, 1, 2).contiguous() for d in output_data]
    outputs = [output_layer(output) for output_layer, output in zip(output_layers, outputs)]
    outputs = torch.cat(outputs, dim=1)
    ids, scores, bboxes = nms_layer(outputs)
    return ids, scores, bboxes

def record_video(filename, duration=300, fps=30, resolution=(128, 160)):
    video_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(video_path):
        print(f"El video '{filename}' ya existe. Se usará el archivo existente.")
        return video_path

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[0])
    cap.set(cv2.CAP_PROP_FPS, fps)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (resolution[1], resolution[0]))

    start_time = time.time()
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (resolution[1], resolution[0]))
            out.write(frame)
            # No mostramos el frame para evitar consumo adicional de recursos
            # cv2.imshow('Recording', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print(f"Video grabado como '{filename}' con resolución {resolution[1]}x{resolution[0]}")
    return video_path

def process_video(video_path, interpreter_path, resolution, result_queue, camera_id):
    # Cargar el intérprete en cada proceso
    interpreter = tf.lite.Interpreter(interpreter_path)
    interpreter.allocate_tensors()

    nms_layer, output_layers = build_det_helper()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    total_detections = 0
    processing_time = 0

    process = psutil.Process(os.getpid())
    cpu_usages = []
    memory_usages = []

    start_time = time.time()  # Inicio del tiempo total

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Inicializar medición de CPU
        process.cpu_percent(interval=None)

        # Medir el tiempo de procesamiento
        process_start = time.time()

        # Procesamiento
        processed_image = preprocess_image(frame, resolution)
        ids, scores, bboxes = eval_image(
            processed_image, interpreter, input_details, output_details, output_layers, nms_layer)

        process_end = time.time()
        processing_time += process_end - process_start

        # Obtener el uso de CPU
        cpu_usage = process.cpu_percent(interval=None)
        cpu_usages.append(cpu_usage)

        # Medir el uso de memoria después del procesamiento
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Convertir a MB
        memory_usages.append(memory_usage)

        threshold = 0.3
        n_positive = (scores > threshold).sum().item()
        total_detections += n_positive

        # No se realiza ninguna operación de visualización

    end_time = time.time()
    total_time = end_time - start_time

    # Calcular métricas
    avg_fps = frame_count / total_time if total_time > 0 else 0
    processing_fps = frame_count / processing_time if processing_time > 0 else 0
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    avg_cpu_usage = np.mean(cpu_usages)
    avg_memory_usage = np.mean(memory_usages)

    cap.release()

    # Colocar los resultados en la cola
    result_queue.put({
        'camera_id': camera_id,
        'avg_fps': avg_fps,
        'processing_fps': processing_fps,
        'avg_detections': avg_detections,
        'avg_cpu_usage': avg_cpu_usage,
        'avg_memory_usage': avg_memory_usage
    })

def main():
    tflite_path = download_tflite(net_id="person-det")

    precision, model_size, num_tensors = get_model_info(tflite_path)

    print(f"La resolución que acepta el modelo es: [128, 160]")
    print(f"Precisión del modelo: {precision}")
    print(f"Tamaño del modelo: {model_size:.2f} MB")
    print(f"Número de tensores en el modelo: {num_tensors}")

    # Grabar o usar video existente
    video_path = record_video('input_video.avi', duration=10, fps=30, resolution=(128, 160))

    results = []

    # Simular de 1 a 10 cámaras (puedes ajustar este número según tu hardware)
    for num_cameras in range(1, 11):
        print(f"\nSimulando {num_cameras} cámaras...")
        processes = []
        result_queue = Queue()

        # Iniciar múltiples procesos
        for cam_id in range(num_cameras):
            p = Process(target=process_video, args=(
                video_path, tflite_path, (128, 160), result_queue, cam_id))
            processes.append(p)
            p.start()

        # Esperar a que todos los procesos terminen
        for p in processes:
            p.join()

        # Recopilar resultados
        total_avg_fps = 0
        total_processing_fps = 0
        total_avg_detections = 0
        total_cpu_usage = 0
        total_memory_usage = 0

        while not result_queue.empty():
            result = result_queue.get()
            total_avg_fps += result['avg_fps']
            total_processing_fps += result['processing_fps']
            total_avg_detections += result['avg_detections']
            total_cpu_usage += result['avg_cpu_usage']
            total_memory_usage += result['avg_memory_usage']

        # Calcular promedios totales
        avg_fps_per_camera = total_avg_fps / num_cameras if num_cameras > 0 else 0
        processing_fps_per_camera = total_processing_fps / num_cameras if num_cameras > 0 else 0
        avg_detections_per_camera = total_avg_detections / num_cameras if num_cameras > 0 else 0
        avg_cpu_usage_per_camera = total_cpu_usage / num_cameras if num_cameras > 0 else 0
        avg_memory_usage_per_camera = total_memory_usage / num_cameras if num_cameras > 0 else 0

        print(f"Promedio de FPS por cámara (incluyendo procesamiento): {avg_fps_per_camera:.2f}")
        print(f"Promedio de FPS de procesamiento por cámara: {processing_fps_per_camera:.2f}")
        print(f"Promedio de detecciones por frame por cámara: {avg_detections_per_camera:.2f}")
        print(f"Uso promedio de CPU por cámara: {avg_cpu_usage_per_camera:.2f}%")
        print(f"Uso promedio de Memoria por cámara: {avg_memory_usage_per_camera:.2f} MB")

        results.append([
            num_cameras,
            avg_fps_per_camera,
            processing_fps_per_camera,
            avg_detections_per_camera,
            avg_cpu_usage_per_camera,
            avg_memory_usage_per_camera
        ])

    # Guardar resultados en CSV
    csv_path = os.path.join(os.getcwd(), 'resultados_simulacion_camaras.csv')
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Número de Cámaras",
            "FPS Promedio por Cámara",
            "FPS Procesamiento por Cámara",
            "Detecciones Promedio por Cámara",
            "Uso Promedio de CPU por Cámara (%)",
            "Uso Promedio de Memoria por Cámara (MB)"
        ])
        writer.writerows(results)

    print(f"\nResultados guardados en '{csv_path}'")

if __name__ == '__main__':
    main()
