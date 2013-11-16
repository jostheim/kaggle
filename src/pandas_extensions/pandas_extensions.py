import pandas as pd
import numpy as np
from pandas.core import frame
from pandas.core.index import MultiIndex
from pandas.tseries.period import PeriodIndex
import pandas.lib as lib
import pandas.core.common as com
import csv
import dateutil
import pytz
from pytz import timezone



def parse_date_time(val):
    if val is not np.nan:
        #'2012-11-12 17:30:00+00:00
        try:
            datetime_obj = dateutil.parser.parse(val)
            datetime_obj = datetime_obj.replace(tzinfo=timezone('UTC'))
            datetime_obj = datetime_obj.astimezone(timezone('UTC'))
            return datetime_obj
        except ValueError as e:
#            print e
            return np.nan
    else:
        return np.nan


def _my_helper_csv(self, writer, na_rep=None, cols=None,
                header=True, index=True,
                index_label=None, float_format=None, write_dtypes=None):
    if cols is None:
        cols = self.columns

    series = {}
    for k, v in self._series.iteritems():
        series[k] = v.values

    has_aliases = isinstance(header, (tuple, list, np.ndarray))
    if has_aliases or header:
        if index:
            # should write something for index label
            if index_label is not False:
                if index_label is None:
                    if isinstance(self.index, MultiIndex):
                        index_label = []
                        for i, name in enumerate(self.index.names):
                            if name is None:
                                name = ''
                            index_label.append(name)
                    else:
                        index_label = self.index.name
                        if index_label is None:
                            index_label = ['']
                        else:
                            index_label = [index_label]
                elif not isinstance(index_label, (list, tuple, np.ndarray)):
                    # given a string for a DF with Index
                    index_label = [index_label]

                encoded_labels = list(index_label)
            else:
                encoded_labels = []

            if has_aliases:
                if len(header) != len(cols):
                    raise ValueError(('Writing %d cols but got %d aliases'
                                      % (len(cols), len(header))))
                else:
                    write_cols = header
            else:
                write_cols = cols
            encoded_cols = list(write_cols)
            if write_dtypes:
                for j, col in enumerate(cols):
                    encoded_cols[j] = "{0}:{1}".format(col, self._series[col].dtype)
            writer.writerow(encoded_labels + encoded_cols)
        else:
            if write_dtypes:
                for j, col in enumerate(cols):
                    encoded_cols[j] = "{0}:{1}".format(col, self._series[col].dtype)
            writer.writerow(encoded_cols)

            encoded_cols = list(cols)
            writer.writerow(encoded_cols)

    data_index = self.index
    if isinstance(self.index, PeriodIndex):
        data_index = self.index.to_timestamp()

    nlevels = getattr(data_index, 'nlevels', 1)
    for j, idx in enumerate(data_index):
        row_fields = []
        if index:
            if nlevels == 1:
                row_fields = [idx]
            else:  # handle MultiIndex
                row_fields = list(idx)
        for i, col in enumerate(cols):
            val = series[col][j]
            if lib.checknull(val):
                val = na_rep

            if float_format is not None and com.is_float(val):
                val = float_format % val
            elif isinstance(val, np.datetime64):
                val = lib.Timestamp(val)._repr_base

            row_fields.append(val)

        writer.writerow(row_fields)

def my_to_csv(self, path_or_buf, sep=",", na_rep='', float_format=None,
               cols=None, header=True, index=True, index_label=None,
               mode='w', nanRep=None, encoding=None, quoting=None,
               line_terminator='\n', write_dtypes=None):
        """
        Write DataFrame to a comma-separated values (csv) file

        Parameters
        ----------
        path_or_buf : string or file handle / StringIO
            File path
        sep : character, default ","
            Field delimiter for the output file.
        na_rep : string, default ''
            Missing data representation
        float_format : string, default None
            Format string for floating point numbers
        cols : sequence, optional
            Columns to write
        header : boolean or list of string, default True
            Write out column names. If a list of string is given it is
            assumed to be aliases for the column names
        index : boolean, default True
            Write row names (index)
        index_label : string or sequence, or False, default None
            Column label for index column(s) if desired. If None is given, and
            `header` and `index` are True, then the index names are used. A
            sequence should be given if the DataFrame uses MultiIndex.  If
            False do not print fields for index names. Use index_label=False
            for easier importing in R
        nanRep : deprecated, use na_rep
        mode : Python write mode, default 'w'
        encoding : string, optional
            a string representing the encoding to use if the contents are
            non-ascii, for python versions prior to 3
        line_terminator: string, default '\n'
            The newline character or character sequence to use in the output
            file
        quoting : optional constant from csv module
            defaults to csv.QUOTE_MINIMAL
        """
        if nanRep is not None:  # pragma: no cover
            import warnings
            warnings.warn("nanRep is deprecated, use na_rep",
                          FutureWarning)
            na_rep = nanRep

        if hasattr(path_or_buf, 'read'):
            f = path_or_buf
            close = False
        else:
            f = com._get_handle(path_or_buf, mode, encoding=encoding)
            close = True

        if quoting is None:
            quoting = csv.QUOTE_MINIMAL

        try:
            if encoding is not None:
                csvout = com.UnicodeWriter(f, lineterminator=line_terminator,
                                           delimiter=sep, encoding=encoding,
                                           quoting=quoting)
            else:
                csvout = csv.writer(f, lineterminator=line_terminator,
                                    delimiter=sep, quoting=quoting)
            self._helper_csv(csvout, na_rep=na_rep,
                             float_format=float_format, cols=cols,
                             header=header, index=index,
                             index_label=index_label, write_dtypes=write_dtypes)

        finally:
            if close:
                f.close()
                
pd.core.frame.DataFrame.to_csv = my_to_csv
pd.core.frame.DataFrame._helper_csv = _my_helper_csv

from pandas.io import parsers

def my_read_csv(filepath_or_buffer, sep=',', dialect=None, compression=None, doublequote=True, escapechar=None, quotechar='"', quoting=0, skipinitialspace=False, lineterminator=None, header='infer', index_col=None, names=None, prefix=None, skiprows=None, skipfooter=None, skip_footer=0, na_values=None, true_values=None, false_values=None, delimiter=None, converters=None, dtype=None, usecols=None, engine='c', delim_whitespace=False, as_recarray=False, na_filter=True, compact_ints=False, use_unsigned=False, low_memory=True, buffer_lines=None, warn_bad_lines=True, error_bad_lines=True, keep_default_na=True, thousands=None, comment=None, decimal='.', parse_dates=False, keep_date_col=False, dayfirst=False, date_parser=None, memory_map=False, nrows=None, iterator=False, chunksize=None, verbose=False, encoding=None, squeeze=False, read_dtypes=None):
    df = pd.read_csv(filepath_or_buffer, sep=sep, dialect=dialect, compression=compression, doublequote=doublequote, escapechar=escapechar, quotechar=quotechar, quoting=quoting, skipinitialspace=skipinitialspace, lineterminator=lineterminator, header=header, index_col=index_col, names=names, prefix=prefix, skiprows=skiprows, skipfooter=skip_footer, skip_footer=skip_footer, na_values=na_values, true_values=true_values, false_values=false_values, delimiter=delimiter, converters=converters, dtype=dtype, usecols=usecols, engine=engine, delim_whitespace=delim_whitespace, as_recarray=as_recarray, na_filter=na_filter, compact_ints=compact_ints, use_unsigned=use_unsigned, low_memory=low_memory, buffer_lines=buffer_lines, warn_bad_lines=warn_bad_lines, error_bad_lines=error_bad_lines, keep_default_na=keep_default_na, thousands=thousands, comment=comment, decimal=decimal, parse_dates=parse_dates, keep_date_col=keep_date_col, dayfirst=dayfirst, date_parser=date_parser, memory_map=memory_map, nrows=nrows, iterator=iterator, chunksize=chunksize, verbose=verbose, encoding=encoding, squeeze=squeeze)
    if read_dtypes:
        for col, series in df.iteritems():
            splits= col.split(":")
            read_dtype = splits[1]
            if str(series.dtype) != read_dtype:
                if read_dtype == "datetime.datetime":
                    series = series.apply(lambda x: parse_date_time(x))
                    series = pd.Series(series.values,dtype='M8[ns]')
                elif "datetime64" in read_dtype:
                    series = series.apply(lambda x: parse_date_time(x))
                    series = pd.Series(df[col].values,dtype='M8[ns]')
                elif read_dtype == "float" or read_dtype == "float64" or read_dtype == "float32": 
                    series.values = series.values.astype(np.float64)
                elif read_dtype == "int" or read_dtype == "int64" or read_dtype == "int32":
                    series.values = series.values.astype(np.int64)
                elif read_dtype == "bool" or read_dtype == "bool_":
                    series.values = series.values.astype(np.bool_)
                elif read_dtype == "complex_":
                    series.values = series.values.astype(np.complex_)
                df[col] = series
    return df
                    
pd.my_read_csv = my_read_csv