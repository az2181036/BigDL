#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from bigdl.nano.common.cpu_schedule import schedule_workers
import os
import json
import shutil
from tempfile import TemporaryDirectory
from contextlib import closing
import socket
import tensorflow as tf


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def train_func(model_dir, ds_graph, elem_spec,
               val_ds_graph, val_elem_sepc, fit_kwargs):
    import tensorflow as tf
    from tensorflow.python.distribute.coordinator.values import deserialize_dataset_from_graph

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        new_model = tf.keras.models.load_model(os.path.join(model_dir, "temp_model"))
        train_dataset = deserialize_dataset_from_graph(ds_graph, elem_spec)
        if val_ds_graph is not None:
            val_dataset = deserialize_dataset_from_graph(val_ds_graph, val_elem_sepc)
        else:
            val_dataset = None

        task_id = strategy.cluster_resolver.task_id

        if task_id == 0:
            verbose = fit_kwargs['verbose']
        else:
            verbose = 0
        del fit_kwargs['verbose']
        history = new_model.fit(train_dataset,
                                validation_data=val_dataset,
                                verbose=verbose,
                                **fit_kwargs)
        if task_id == 0:
            path = os.path.join(model_dir, 'trained_model_weights')
            new_model.save_weights(path, overwrite=True)
        else:
            path = os.path.join(model_dir, f'trained_model_weights_{task_id}')
            new_model.save_weights(path, overwrite=True)
        return history


def train_func_env(model_dir, ds_graph, elem_spec,
               val_ds_graph, val_elem_sepc, fit_kwargs, env):
    env_back, env_del_list = dict(), list()
    for key, value in env.items():
        if key in os.environ:
            env_back[key] = os.environ[key]
        else:
            env_del_list.append(key)
        os.environ[key] = value
    import tensorflow as tf
    from tensorflow.python.distribute.coordinator.values import deserialize_dataset_from_graph

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        new_model = tf.keras.models.load_model(os.path.join(model_dir, "temp_model"))
        train_dataset = deserialize_dataset_from_graph(ds_graph, elem_spec)
        if val_ds_graph is not None:
            val_dataset = deserialize_dataset_from_graph(val_ds_graph, val_elem_sepc)
        else:
            val_dataset = None

        task_id = strategy.cluster_resolver.task_id

        if task_id == 0:
            verbose = fit_kwargs['verbose']
        else:
            verbose = 0
        del fit_kwargs['verbose']
        history = new_model.fit(train_dataset,
                                validation_data=val_dataset,
                                verbose=verbose,
                                **fit_kwargs)
        if task_id == 0:
            path = os.path.join(model_dir, 'trained_model_weights')
            new_model.save_weights(path, overwrite=True)
        else:
            path = os.path.join(model_dir, f'trained_model_weights_{task_id}')
            new_model.save_weights(path, overwrite=True)

        for key, value in env_back.items():
            os.environ[key] = value
        for _, key in enumerate(env_del_list):
            del os.environ[key]
        return history


def distributed_train_keras(backend, model, nprocs, fit_kwargs=None):

    backend.setup()

    if fit_kwargs is None:
        fit_kwargs = {}

    cpu_procs = schedule_workers(nprocs)

    from tensorflow.python.distribute.coordinator.values import serialize_dataset_to_graph

    train_dataset = fit_kwargs.pop('x')
    val_dataset = fit_kwargs.pop('validation_data')

    train_ds_def = serialize_dataset_to_graph(train_dataset).numpy()
    train_elem_spec = train_dataset.element_spec

    if val_dataset is not None:
        val_ds_def = serialize_dataset_to_graph(val_dataset).numpy()
        val_elem_spec = val_dataset.element_spec
    else:
        val_ds_def = None
        val_elem_spec = None

    # this is to work around a tensorflow bug: https://github.com/keras-team/keras/issues/16023
    model.evaluate(train_dataset, verbose=0, steps=1)
    assert model.compiled_metrics.built

    ports = set()
    while len(ports) < nprocs:
        ports.add(find_free_port())
    ports = list(ports)
    worker_list = [f"localhost:{p}" for p in ports]

    with TemporaryDirectory() as temp_dir:
        model.save(os.path.join(temp_dir, 'temp_model'))

        envs = []
        for i in range(nprocs):
            env = {
                "KMP_AFFINITY": f"granularity=fine,proclist"
                                f"=[{','.join([str(i) for i in cpu_procs[i]])}],explicit",
                "OMP_NUM_THREADS": str(len(cpu_procs[i])),
                "TF_CONFIG": json.dumps(
                    {
                        'cluster': {
                            'worker': worker_list
                        },
                        'task': {'type': 'worker', 'index': i}
                    }),
                'no_proxy': "localhost",
            }
            envs.append(env)

        train_args = (temp_dir, train_ds_def, train_elem_spec,
                      val_ds_def, val_elem_spec, fit_kwargs)

        histrories = backend.run_process(target=train_func, args=train_args, nprocs=nprocs, envs=envs)
        # histrories = backend.run_pool(target=train_func_env, args=train_args, nprocs=nprocs, envs=envs)
        model.load_weights(os.path.join(temp_dir, 'trained_model_weights'))
    return histrories[0]


def train_func_test(model_dir, envs=...):
    import tensorflow as tf
    from tensorflow.python.distribute.coordinator.values import deserialize_dataset_from_graph

    for key, value in envs.items():
        os.environ[key] = value
    print(os.environ['TF_CONFIG'])
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential

        import pathlib
        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
        data_dir = pathlib.Path(data_dir)

        batch_size = 32
        img_height = 180
        img_width = 180

        train_dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        # new_model = tf.keras.models.load_model(os.path.join(model_dir, "temp_model"))

        class_names = train_dataset.class_names

        AUTOTUNE = tf.data.AUTOTUNE

        train_dataset = train_dataset.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)

        num_classes = len(class_names)

        model = Sequential([
            layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        task_id = strategy.cluster_resolver.task_id

        fit_kwargs = dict(epochs=3, verbose='auto')

        if task_id == 0:
            verbose = fit_kwargs['verbose']
        else:
            verbose = 0
        del fit_kwargs['verbose']
        history = model.fit(train_dataset,
                                validation_data=val_dataset,
                                verbose=verbose,
                                **fit_kwargs)
        if task_id == 0:
            path = os.path.join(model_dir, 'trained_model_weights')
            model.save_weights(path, overwrite=True)
        else:
            path = os.path.join(model_dir, f'trained_model_weights_{task_id}')
            model.save_weights(path, overwrite=True)
        return history


def distributed_train(backend, nprocs):

    backend.setup()

    cpu_procs = schedule_workers(nprocs)
    ports = set()
    while len(ports) < nprocs:
        ports.add(find_free_port())
    ports = list(ports)
    worker_list = [f"localhost:{p}" for p in ports]

    with TemporaryDirectory() as temp_dir:

        envs = []
        for i in range(nprocs):
            env = {
                "KMP_AFFINITY": f"granularity=fine,proclist"
                                f"=[{','.join([str(i) for i in cpu_procs[i]])}],explicit",
                "OMP_NUM_THREADS": str(len(cpu_procs[i])),
                "TF_CONFIG": json.dumps(
                    {
                        'cluster': {
                            'worker': worker_list
                        },
                        'task': {'type': 'worker', 'index': i}
                    }),
                'no_proxy': "localhost",
            }
            envs.append(env)

        histrories = backend.run(target=train_func_test,
                                 nprocs=nprocs, args=temp_dir,
                                 envs=envs)


if __name__ == "__main__":
    from bigdl.nano.common.multiprocessing.multiprocs_backend import MultiprocessingBackend
    from bigdl.nano.common.multiprocessing.ray_backend import RayBackend
    import time

    st = time.time()
    distributed_train(MultiprocessingBackend(), nprocs=2)
    print(time.time()-st)
