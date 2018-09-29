import gdal
import numpy as np


class RSImage(object):
    def __init__(self, file_path):
        self.img_path = file_path
        self.img_metaInfo = None
        self.projection = None
        self.dataTypeName = None
        self.geoTransform = None
        self.bandCount = 1

        self.dataset = None
        self.img_arr = None

        self.read_info()

        self.raster_X = self.dataset.RasterXSize
        self.raster_Y = self.dataset.RasterYSize
        self.bandCount = self.dataset.RasterCount

        self.read_data()

    def read_info(self):
        self.dataset = gdal.Open(self.img_path)
        self.img_metaInfo = self.dataset.GetMetadata()
        self.projection = self.dataset.GetProjection()
        self.geoTransform = self.dataset.GetGeoTransform()

    def read_data(self):
        self.img_arr = np.zeros((self.raster_Y, self.raster_X,
                                 self.bandCount), 'uint8')

        for i in range(self.bandCount):
            self.img_arr[..., i] = self.dataset.GetRasterBand(i + 1).ReadAsArray()

    def save(self, dst_filename, input_arr):
        geotransform = self.geoTransform
        geoprojection = self.projection

        driver = self.dataset.GetDriver()

        dst_ds = driver.Create(dst_filename, xsize=self.raster_X, ysize=self.raster_Y,
                               bands=self.bandCount, eType=gdal.GDT_Byte)

        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(geoprojection)

        for i in range(self.bandCount):
            # read the data of one band
            raster = input_arr[:, :, i]

            dst_ds.GetRasterBand(i+1).WriteArray(raster)
            print("band " + str(i + 1) + " has been processed")


def unit_test():
    rsObj = RSImage('./data/nudt2017-08-18/nudt2017-08-18.tif')
    print(rsObj.img_metaInfo)

    print(type(rsObj.img_arr))
    print(rsObj.img_arr.shape)
    print(rsObj.dataTypeName)

    rsObj.save('./data/save.tif', rsObj.img_arr)


if __name__ == '__main__':
    unit_test()
