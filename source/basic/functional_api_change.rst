.. _functional_api_change:

Functional API 变更与说明
==============================

本章节列出了 v0.6 与 v1.0 版本的 Functional API 对比，并以 v1.0 为基准（中间列）。
有些 API 是 v1.0 版本新增的，表现为 v0.6 列为空。有些 API 在 v1.0 中被删除了，表现为 v1.0 列为空。

.. note::
   红色部分API无法直接通过 `F.xxx` 调用，需要通过 `F.nn.xxx` 来调用。

.. tabularcolumns:: |p{5cm}|p{5cm}|p{14cm}|p{14cm}|

.. csv-table:: API 对应关系表以及相关说明
   :file: attachs/api_change.csv
   :header-rows: 1
   :class: longtable
   :widths: 1 1 8 8
