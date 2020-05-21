"""Arguments for configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import six
import argparse


def str2bool(v):
    """ because argparse does not support to parse "true, False" as python boolean directly"""
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    """ define args """
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, positional_arg=False, **kwargs):
        """ trick code for argument true, false"""
        prefix = "" if positional_arg else "--"
        type = str2bool if type == bool else type
        self._group.add_argument(
            prefix + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def print_arguments(args):
    """ pretty print args """
    logging.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        logging.info('%s: %s' % (arg, value))
    logging.info('------------------------------------------------')
