import tensorflow as tf
import model
from icdar import restore_rectangle
import lanms
import os


def main(export_path='/tmp/east/1'):
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_state = tf.train.get_checkpoint_state('./east_icdar2015_resnet_v1_50_rbox')
    model_path = os.path.join('east_icdar2015_resnet_v1_50_rbox', os.path.basename(ckpt_state.model_checkpoint_path))
    saver.restore(sess, model_path)

    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={
            'input_data': input_images,
        },
        outputs={
            'score': f_score,
            'geometry': f_geometry,
        },
    )

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        print(main(sys.argv[1]))
    else:
        print(main())
