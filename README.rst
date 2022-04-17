# bbpn
Bye Bye Pink Noise (BBPN)

1/f noise reduction tool for JWST cal.fits images.


Demonstration
~~~~~~~~~~~~~

![Alt Text](./demo.gif)

![](./demo.gif)

.. image:: ./demo.gif


Examples
~~~~~~~~
.. code-block:: python
    from bbpn import bbpn
    bbpn.run(cal_file)


Arguments
~~~~~~~~~
- file_seg: Segmantation mask image for the input cal.fits file. If None, the module will try to find _seg.fits. Default None.

- plot_res: Plot results from each step. Defaule False. 

- file_out: Output file name. Default None.

- f_write: Flag to write the output fits file. Default True.

