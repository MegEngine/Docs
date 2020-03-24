from typing import Tuple, Union


class ExampleClass:
    r"""brief information here

    A new paragraph need a blank line break. If docstring contains backslashes
    then add 'r' at the beginning.

    Indent examples:
        1. simple list(indent is 2! need blank line between father and children list):

        Args:

        * item1

          * subitem1

        * item2
        * item3

        2. definition list(no blank line between head and content):

        Args:
            item1

            item2

            item3

        3. block quotes(need a blank line between head and content):

        Args:

            item1

            item2

            item3

    Math examples:
        1. single line math equation:

        .. math::

            Z_z = \sum_{(x, y) \in F_z} X_x Y_y

        2. inline math symbols:

        We can denote :math:`F` by :math:`(x,y)\mapsto z`, then
        :math:`R_1((x,y)\mapsto z) = (y, z) \mapsto x`, and
        :math:`R_2((x,y)\mapsto z) = (z, x) \mapsto y`; it follows
        :math:`R_1(R_2((x,y)\mapsto z)) = R_2(R_1((x,y)\mapsto z)) =
        (x,y)\mapsto z`.

    Note examples:
        .. note::

            ``W`` and ``b`` can be provided in the ``kwargs`` to specify the filter
            and bias.

    Cross reference examples:
        1. class, attribute, examples:

        class example is :class:`~.DataType`. attribute example is :attr:`~.ExampleClass.data_type`.

        2. reference example:

        you need to define a reference ahead of a title first and then reference it.
        see :ref:`example_reference` for more details.

    Code examples(https://www.sphinx-doc.org/en/1.5/markup/code.html):
        1. default code rendered as python code(remember the blank line and indent):

        ::

            import numpy
            print(numpy.random.normal(size=(10, 10)))

        2. show other languages' code:

        .. code-block:: bash

            cd $(dirname $0)
            rm -rf build

        3. add ``caption`` and reference it using ``name``:

        .. code-block::
            :caption: this.py
            :name: this-py

            print 'Explicit is better than implicit.'

        4. include other source file(path is relative to your rst file rather than py file):

        .. literalinclude:: ../conf.py
            :language: python
            :lines: 20-22

    Doctest examples(https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html):
        1. use ``test_setup`` to import some packages or define some variables which will be hidden in built pages.

        .. testsetup::

            import datetime

        2. use ``testcode`` and ``testoutput`` for complete code block test.

        .. testcode::

            datetime.date(1994, 11, 1)         # this will give no output!
            print(datetime.date(1994, 11, 1))  # this will give output

        .. testoutput::

            1994-11-01

        3. or use ``doctest`` for interactive code block.ip

        .. doctest::

            >>> import datetime
            >>> datetime.date(1994, 11, 1)
            datetime.date(1994, 11, 1)

    """
    class DataType:
        FLOAT = "FLOAT"
        """
        input/output both float32/float16
        """
        INT8x8x16 = "INT8x8x16"
        INT8x8x32 = "INT8x8x32"
        FLOAT_IO16xC32 = "FLOAT_IO16xC32"
        """
        input/output both float16, the internal compute is float32
        """
        QUINT8x8x32 = "QUINT8x8x32"
        """
        input QuantizedAsymm8, output QuantizedS32
        """
        INT8x8xX = "INT8x8xX"
        """
        input int8, output specified by tensor DType
        """
        QUINT4x4x32 = "QUINT4x4x32"
        """
        input QuantizedAsymm4, output QuantizedS32
        """

    _meta_data_type_type = DataType

    __hyperparam_spec__ = (
        ('data_type', 'cvt', _meta_data_type_type),
        ('dilate_shape', 'cvt', _meta_data_type_type),
        ('compute_mode', 'cvt', _meta_data_type_type),
    )

    data_type = _meta_data_type_type.FLOAT
    """input/output data type"""

    dilate_shape = (1, 1)

    _group_spec = None

    def __init__(self, params: dict):
        pass

    def example_func(
            self,
            kernel_shape: Union[int, Tuple[int, int]] = None,
            output_nr_channel: int = None,
            group: Union[int, str] = None,
            **kwargs
    ):
        r"""a brief function description here.

        :param kernel_shape: shape of the convolution kernel; it can be omitted
            only when *W* is given as a :class:`.VarNode`
        :param output_nr_channel: total numebr of channels for output; it can
            be omitted only when *W* is given as a :class:`.VarNode`
        :param group: divide the input, output and filter tensors into groups
            to form a sparse connection; in such case, the filter would have an
            extra first dimension as the number of groups, and filter layout is
            ``[group, output_channel_per_group, input_channel_per_group,
            spatial_dims...]``. Valid values include:

            * ``None``: Do not use grouped convolution;
            * an ``int`` value: Specify the number of groups directly;
            * ``'chan'``: Channel-wise convolution: number of groups equals
              to number of channels in the input tensor. In such case,
              ``output_nr_channel`` can be omitted, and it would be set to
              number of input channels if it is indeed ommitted.
        """
        self._group_spec = group

        super().__init__(kernel_shape, output_nr_channel, **kwargs)
