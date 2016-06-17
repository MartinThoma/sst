#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Start a webserver which can record the data and work as a classifier."""

import logging
import os
import scipy
import sys
import time
import uuid

from flask import Flask, request, render_template, url_for
from flask_bootstrap import Bootstrap
from pkg_resources import resource_filename

try:
    from urllib import parse as urlquote
except ImportError:  # Python 2 fallback
    from urllib import quote as urlquote

# Custom modules
from . import utils
from .eval_image import eval_net

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

# configuration
DEBUG = True


template_path = resource_filename('sst', 'templates/')
logging.info('template_path: %s', template_path)
app = Flask(__name__,
            template_folder=template_path,
            static_path=resource_filename('sst', 'static/'))
Bootstrap(app)
app.config.from_object(__name__)
nn = None
hypes = None


@app.route('/', methods=['POST', 'GET'])
def index():
    """Start page."""
    return (('<a href="work?photo_path=%s&stride=100">'
             'Classify </a>') %
            urlquote(utils.get_default_data_image_path()))


@app.route('/work', methods=['POST', 'GET'])
def work():
    """A worker task."""
    global nn
    if request.method == 'GET':
        out_filename = os.path.join(resource_filename('sst', 'static/'),
                                    'out-%s.png' % uuid.uuid4())

        photo_path = request.args.get('photo_path',
                                      utils.get_default_data_image_path())
        patch_size = hypes['segmenter']['patch_size']
        hard_classification = request.args.get('hard_classification',
                                               '0') == '1'
        output_path = request.args.get('output_path', out_filename)
        stride = min(int(request.args.get('stride', 10)), patch_size)
        logging.info("photo_path: %s", photo_path)
        t0 = time.time()
        logging.info('photo_path: %s', photo_path)
        logging.info('parameters: %s', hypes)
        result = eval_net(hypes=hypes,
                          trained=nn,
                          photo_path=photo_path,
                          stride=stride,
                          hard_classification=hard_classification)
        scipy.misc.imsave(output_path, result)
        t1 = time.time()
        overlay_path = utils.overlay_images(photo_path,
                                            result,
                                            utils.get_overlay_name(output_path))
        logging.info("output_path: %s", output_path)
        logging.info("Overlay path: %s", overlay_path)
        t2 = time.time()

        # Get image files
        image_paths = []
        if 'DATA' in os.environ:
            for path, subdirs, files in os.walk(os.environ['DATA']):
                for name in files:
                    if name.endswith('.png'):
                        image_paths.append(os.path.join(path, name))

        return render_template('canvas.html',
                               execution_time=t1 - t0,
                               overlay_time=t2 - t1,
                               photo_path=photo_path,
                               patch_size=patch_size,
                               stride=stride,
                               output_path=url_for('static',
                                                   filename=os.path.basename(output_path)),
                               output_overlay_path=url_for('static',
                                                           filename=os.path.basename(utils.get_overlay_name(output_path))),
                               hard_classification=hard_classification,
                               image_paths=[])
    else:
        return "Request method: %s" % request.method


def get_parser():
    """Return the parser object for this script."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hypes",
                        dest="hypes_file",
                        type=str,
                        required=True,
                        help=("path to a JSON file with "
                              "contains 'data' (with 'train' and 'test') as "
                              "well as 'classes' (with 'colors' for each)"))
    parser.add_argument("--port",
                        dest="port", default=5000, type=int,
                        help="where should the webserver run")
    return parser


def main(hypes_file, port):
    """Main function starting the webserver."""
    global nn, hypes
    hypes = utils.load_hypes(hypes_file)
    if nn is None:
        nn = utils.deserialize_model(hypes)
    logging.info("Start webserver...")
    app.run(port=port)

if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.hypes_file, args.port)
