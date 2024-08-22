import numpy as np
import cv2


class Draw:
    @staticmethod
    def object_id(image, object):
        if object.label is not None and object.confidence is not None:
            label = f"#{object.object_id}: {object.confidence} {object.label}"
        else:
            label = f"#{object.object_id}"

        border_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 25)
        label_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.putText(image, label,
                    (np.max([0, object.centroid[0] - border_length[0][0] // 2 + 10]),
                     np.max([0, object.centroid[1] - 30])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple([i + 30 for i in object.color]), 25, cv2.LINE_AA)
        cv2.putText(image, label,
                    (np.max([0, object.centroid[0] - label_length[0][0] // 2]),
                     np.max([0, object.centroid[1] - 30])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple([i - 30 for i in object.color]), 2, cv2.LINE_AA)

    @staticmethod
    def bounding_box(image, object, radius=20, length=15, thickness=4):
        if object.box is not None:
            x_start, y_start = object.box[0], object.box[1]
            x_end, y_end = object.box[2], object.box[3]
            cv2.line(image, (x_start + radius, y_start), (x_start + radius + length, y_start), object.color, thickness)
            cv2.line(image, (x_start, y_start + radius), (x_start, y_start + radius + length), object.color, thickness)
            cv2.ellipse(image, (x_start + radius, y_start + radius), (radius, radius), 180, 0, 90, object.color,
                        thickness)
            cv2.line(image, (x_end - radius, y_start), (x_end - radius - length, y_start), object.color, thickness)
            cv2.line(image, (x_end, y_start + radius), (x_end, y_start + radius + length), object.color, thickness)
            cv2.ellipse(image, (x_end - radius, y_start + radius), (radius, radius), 270, 0, 90, object.color,
                        thickness)
            cv2.line(image, (x_start + radius, y_end), (x_start + radius + length, y_end), object.color, thickness)
            cv2.line(image, (x_start, y_end - radius), (x_start, y_end - radius - length), object.color, thickness)
            cv2.ellipse(image, (x_start + radius, y_end - radius), (radius, radius), 90, 0, 90, object.color, thickness)
            cv2.line(image, (x_end - radius, y_end), (x_end - radius - length, y_end), object.color, thickness)
            cv2.line(image, (x_end, y_end - radius), (x_end, y_end - radius - length), object.color, thickness)
            cv2.ellipse(image, (x_end - radius, y_end - radius), (radius, radius), 0, 0, 90, object.color, thickness)

    @staticmethod
    def scene(image, object, scene_width=400, scene_height=300, thickness=4):
        size = object.box[3] - object.box[1]

        if object.box is not None and size > scene_height:
            coefficient = size / scene_height
            margin = 50

            start = (
                np.max([0, object.centroid[0] - int(scene_width // 2 * coefficient) - margin]),
                np.max([0, object.box[1] - margin]))
            end = (np.min([image.shape[1] - 1, object.centroid[0] + int(scene_width // 2 * coefficient) + margin]),
                   np.min([image.shape[0] - 1, object.box[3] + margin]))
        else:
            start = (
                np.max([0, object.centroid[0] - scene_width // 2]), np.max([0, object.centroid[1] - scene_height // 2]))
            end = (np.min([image.shape[1] - 1, object.centroid[0] + scene_width // 2]),
                   np.min([image.shape[0] - 1, object.centroid[1] + scene_height // 2]))

        cv2.rectangle(image, start, end, object.color, thickness)

    @staticmethod
    def actions(image, object, actions, margin=150):
        thickness_out = 25
        thickness_in = 2
        size = 0.6
        label = " | ".join(actions)
        border_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, thickness_out)
        label_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, thickness_in)
        centroid = object.centroid

        cv2.putText(image, label,
                    (np.max([0, centroid[0] - border_length[0][0] // 2 + 10]),
                     np.max([0, centroid[1] + margin])),
                    cv2.FONT_HERSHEY_SIMPLEX, size, tuple([i + 100 for i in object.color]), thickness_out, cv2.LINE_AA)
        cv2.putText(image, label,
                    (np.max([0, centroid[0] - label_length[0][0] // 2]),
                     np.max([0, centroid[1] + margin])),
                    cv2.FONT_HERSHEY_SIMPLEX, size, tuple([i - 100 for i in object.color]), thickness_in, cv2.LINE_AA)

    @staticmethod
    def track(image, centroids, color, history):
        start = np.max([1, len(centroids) - history])
        faded = tuple([i - 30 for i in color])
        cv2.circle(image, centroids[-1], 10, faded, -1)

        for i in range(start, len(centroids)):
            thickness = int(np.sqrt(64 * float(i - start + 1)) / 5)
            cv2.line(image, tuple(centroids[i - 1]), tuple(centroids[i]), color, thickness)
