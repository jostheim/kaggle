    '''
    Created on Feb 18, 2013
    
    @author: jostheim
    '''
    import unittest
    import pandas as pd
    import numpy as np
    import datetime
    import pandas_extensions
    
    class Test(unittest.TestCase):
    
    
        def test_reading_and_writing(self):
            df = pd.DataFrame({'a':[1,2,4,7], 'b':[1.2, 2.3, 5.1, 6.3], 
                        'c':list('abcd'), 
                        'd':[datetime.datetime(2001,1,1),datetime.datetime(2001,1,2),np.nan, datetime.datetime(2012,11,2)] })
            for col, series in df.iteritems():
                df['d'] = pd.Series(df['d'].values,dtype='M8[ns]')
            df.to_csv("/tmp/test.csv", write_dtypes=True)
            new_df = pd.my_read_csv("/tmp/test.csv", index_col=0, read_dtypes=True)
            for i, t in enumerate(df.dtypes):
                print t, new_df.dtypes[i]
                self.assertEqual(t, new_df.dtypes[i], "dtypes match")
    
    
    if __name__ == "__main__":
        #import sys;sys.argv = ['', 'Test.test_reading_and_writing']
        unittest.main()