import numpy as np


class Object:
    def __init__(self, object_id, centroid, color, attribute=None):
        self.object_id = object_id
        self.centroid = centroid
        self.color = color
        self.attribute = attribute

    def __getattr__(self, name):
        if self.attribute is None:
            return None

        if self.attribute.get(name) is None:
            return None
        else:
            return self.attribute[name]

    @staticmethod
    def object_factory(objects, centroids, colors, attributes):
        entities = []

        for object_id, centroid in objects.items():
            assigned = False

            for i, c in enumerate(centroids):
                if np.array_equal(centroid, c):
                    entities.append(Object(object_id, centroid, colors[object_id], attributes[i]))
                    assigned = True

            if not assigned:
                # Object disappeared for some frames.
                entities.append(Object(object_id, centroid, colors[object_id], None))

        return entities
