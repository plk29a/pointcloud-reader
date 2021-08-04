import os, sys
import psutil

import laspy
import pandas as pd
import numpy as np

from shapely.geometry import Polygon, Point, LineString, box
from multiprocessing import Pool, cpu_count
from itertools import repeat
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt


class ToSmallCloud(Exception):
    pass

def show(A:np.array):
    """Wyświetlenie chmury punktów w 3 rzutach"""
    plt.scatter(A[:, 0], A[:, 1], alpha=0.5)
    plt.show()

    if A.shape[-1] > 1:
        plt.scatter(A[:, 1], A[:, 2], alpha=0.5)
        plt.show()

        plt.scatter(A[:, 0], A[:, 2], alpha=0.5)
        plt.show()


class Lasek:
    """Klasa do zarzadzania chmura punktow przy wykorzystaniu biblioteki numpy"""
    def __init__(self, path_cloud):
        self.path = path_cloud
        self.name = os.path.split(path_cloud)[-1]

        self.setup()
        if os.stat(path_cloud).st_size <10000:
            raise ToSmallCloud

    def setup(self):
        with laspy.open(self.path) as las:
            self.head = las.header

            Xmin, Xmax, Ymin, Ymax = self.head.x_min, self.head.x_max, self.head.y_min, self.head.y_max
            bbox_coords = (Xmin, Xmax, Ymin, Ymax)
            self.bbox = pd.DataFrame([[Xmin, Ymin], [Xmax, Ymax]], columns=['X' ,'Y'], index=['min', 'max'])
            self.bbox_poligon_offset = box(Xmin-5, Ymin-5, Xmax+5, Ymax+5)
            self.points_count = self.head.point_count
            # self.read_np_array(las)

    def read_np_array(self, chunk_size:int = 50_000_000):
        """Wczytanie chmury punktów jako np array
        W przypadku dużych plików wczytanie chunkami, możliwe późniejsze wprowadzenie generatora"""
        with laspy.open(self.path) as las:
            las_np = np.zeros((0,6))
            if chunk := (self.points_count > chunk_size):
                pbar1 = tqdm(total=(self.points_count // chunk_size + 1), leave=False)
                coment = lambda x: f'Wczytywanie: {self.name[:20]}...{self.name[-8:]}, RAM{x}'
                pbar1.set_description(coment(psutil.virtual_memory().percent))
            for lasek in las.chunk_iterator(chunk_size):
                lasek_np = np.vstack((lasek.x, lasek.y, lasek.z, lasek.red, lasek.green, lasek.blue)).transpose()
                las_np = np.concatenate((las_np, lasek_np))
                if chunk:
                    pbar1.update(1)
                    pbar1.set_description(coment(psutil.virtual_memory().percent))
            tqdm.write(f'Wczytano {self.name}, wykorzystanie ram {psutil.virtual_memory().percent}%')
            if chunk:
                pbar1.close()
            # print(f'Wczytano {self.name}, wykorzystanie ram {psutil.virtual_memory().percent}%\t\t')
            self.las_np = las_np
            return self.las_np

    def point_is_in(self, XY):
        """Sprawdzenie czy punkt zawiera się w chmurze"""
        pkt = Point(XY)
        test = self.bbox_poligon_offset.contains(pkt)
        return test

    def policz_test_sprawdzenia(self):
        punkt = np.array([650620.843 , 436035.3973])
        coords2 = np.delete(self.las_np, -1, axis=1)

        x = datetime.now()
        print(x, psutil.virtual_memory())
        distances = np.linalg.norm((coords2 - punkt), axis=1)
        print(datetime.now() - x , ', ', psutil.virtual_memory())

    @staticmethod
    def save_pc_las(las_np: np.array, path:str, save_new=False):
        """przy zapisie tablica numpy musi zawierac XYZ kolejne kolumny będą interpretowane jako RGB"""
        new_hdr = laspy.LasHeader()
        new_las = laspy.LasData(new_hdr)
        new_las.X = las_np[:, 0] * 100
        new_las.Y = las_np[:, 1] * 100
        new_las.Z = las_np[:, 2] * 100
        if las_np.shape[-1] > 3:
            new_las.red = las_np[:, 3]
            new_las.green = las_np[:, 4]
            new_las.blue = las_np[:, 5]

        try:
            if os.path.exists(path) and not save_new:
                with laspy.open(path, mode="a") as writer:
                    writer.append_points(new_las.points)
            else:
                new_las.write(path)
        except Exception as e:
            print(e, ' nie udalo sie zapisac pliku')

    @staticmethod
    def save_pc_txt(las_np: np.array, path):

        fmt = ['%0.3f', '%0.3f', '%0.3f', '%0.0f', '%0.0f', '%0.0f']
        header = ['X', 'Y', 'Z', 'R', 'G', 'B']
        columns = las_np.shape[-1]
        fmt = ','.join(fmt[:columns])
        header = ','.join(header[:columns])

        las_np[:, 3:6] = las_np[:, 3:6] / 256

        if os.path.isfile(path):
            header = ''

        with open(path, "a") as f:
            np.savetxt(f, las_np, fmt=fmt, header=header, comments='')  # , newline=f'\n{1}, header='Nr X Y Z R G B''

    def __del__(self):
        try:
            del self.las_np
        except:
            pass


class Layerek:
    """Klasa do zarzadzania warstwa wektorową"""
    def __init__(self, path):
        self.path = path
        self.shp_df = self.read_to_df()

    def read_to_df(self):
        """Wczytanie warstwy z punktami XY dla slupow trakcyjnych"""
        ara = pd.read_csv(self.path).set_index('Nr')

        NNN = pd.DataFrame(columns=['XY', 'XY2'])

        NNN['XY'] = ara[['X1', 'Y1']].apply(lambda x: tuple(x), axis=1)
        NNN['XY2'] = ara[['X2', 'Y2']].apply(lambda x: tuple(x), axis=1)

        return NNN

class Lasek_clip(Lasek):
    """Klasa do tworzenia wycinka i zarzadzania wycinkiem calej chmury punktow
    cutted - okresla czy chmura wczytana jest juz przycieta czy nalezy ja przyciac
    point - jesy wierszem tablicy pd z Layerek"""

    def __init__(self, las_np:np.array, point:pd.DataFrame, cutted:bool=False):
        if not cutted:
            self.las_np = self.clip_circle(las_np, point)
        else:
            self.las_np = las_np

        if len(self.las_np) == 0:
            raise ToSmallCloud

        self.point = point

    @staticmethod
    def clip_circle(las_np, point):
        """Przycina chmure punktow okręgiem o długości 5m do punktu wstawienia obiektu"""
        distances = np.linalg.norm((las_np[:,0:2] - point[1][0]), axis=1)
        mask = distances < 5
        np_las_clip = las_np[mask]
        return np_las_clip

    def calc_stats(self, obiect_hight:float=9, offset:float=0.2):
        """Obliczanie statystyki chmury na potrzeby wycinania slupa trakcyjnego, pozbycia się szumu pod chmurą."""
        self.las_np_nocolor = self.las_np[:, 0:3]
        self.point_insert = self.point.loc['XY']
        self.point_index = self.point.name

        Z_mean = self.las_np_nocolor.mean(axis=0)[-1]
        Z_max = self.las_np_nocolor.max(axis=0)[-1]
        Z_min = self.las_np_nocolor.min(axis=0)[-1]
        # assert (Z_max - Z_min) < 20, "cos dziwnie duzo szumu"
        Z_median = np.median(self.las_np_nocolor, axis=0)[-1]
        if (Z_max - Z_min) > 15:
            Z_max = Z_mean + 10
        self.Z_my_mean = (Z_max + Z_median) / 2
        point_insert_z = Z_median

        point_insert = np.append(self.point_insert, point_insert_z)
        point_insert_top = np.append(self.point_insert, point_insert_z+obiect_hight)

        # dodanie offsetu dla punktów wstawienia dla latarni które się nie złożyły
        point_insertN = point_insert[:] + [offset, 0 ,0]
        point_insertS = point_insert[:] + [-offset, 0 ,0]
        point_insertW = point_insert[:] + [0, offset ,0]
        point_insertE = point_insert[:] + [0, -offset ,0]
        point_insertD = point_insert[:] + [0, 0, -offset]
        list_od_point = [point_insert, point_insert_top, point_insertN, point_insertS, point_insertW, point_insertE, point_insertD]

        # dodanie punktow dla pionu co 10cm
        array_main = []
        for x in np.linspace(0,obiect_hight,obiect_hight*10):
            point_insert_main = point_insert[:] + [0, 0, x]
            array_main.append(point_insert_main)
        self.distinctive_point = np.array(list_od_point+array_main)

        # dodaje punkt nad torami
        try:
            self.point_offset = self.point.loc['XY2']
            point_offset = np.append(self.point_offset, point_insert_z+obiect_hight)
            self.distinctive_point = np.array(list_od_point+point_offset+array_main)
        except Exception as e:
            pass

    def clip_to_box(self):
        self.calc_stats()
        # Ucina ziemie z chmury punktów
        las_top = self.las_np_nocolor[self.las_np_nocolor[:, -1] > self.Z_my_mean]
        las_top = las_top[las_top[:, -1] < self.Z_my_mean+8]
        # Stworzenie poligonu do przycięcia
        poligon = self.fit_poligon()
        # Wybranie punktów nalerzacych do poligonu
        las_top = self.cut_to_poligon(las_top, poligon)
        # Przerzedzenie do 500 pkt
        las_top = self.dwindle(las_top, how_much=500)
        # Wklejenie do tablicy punktu wstawienia, oraz punkt końca slupa czyli punktów charakterystycznych

        las_top = np.vstack([las_top, self.distinctive_point])
        # self.point_offset

        return las_top

    @staticmethod
    def cut_to_poligon(las_np:np.array, poligon):
        mask = np.apply_along_axis(lambda x: poligon.contains(Point(x)), 1, las_np[:, [0, 1]])
        return las_np[mask]

    def fit_poligon(self, buffer = 0.3):
        # Stworzenie poligonu do bardziej dokladnego wyciecia chmury
        latarnia_line = LineString([self.point_insert[0:2], self.point_offset])
        latarnia_poligon = latarnia_line.buffer(buffer)
        return latarnia_poligon

    @staticmethod
    def dwindle(las_np:np.array, how_much = 1000):
        number_of_rows = las_np.shape[0]

        size = how_much
        if number_of_rows <= 1000:
            size = number_of_rows

        random_indices = np.random.choice(number_of_rows, size=size, replace=True)

        las_np = las_np[random_indices, :]
        return las_np


class PChandler:
    """Do wycinanaia i zapisywania elemntów z wielu chmur punktów"""
    def __init__(self, dir_to_las:str, dir_to_save:str, shp_path:str):
        self.path = dir_to_las
        self.path_save = dir_to_save

        self.shp_path = shp_path
        y = Layerek(self.shp_path)
        self.lista_points = y.shp_df

        self.list_of_pc = self.get_list_check(dir_to_las, '.las')

        #Stworzenie listy obiektów:
        wector_layer = Layerek(shp_path)
        self.element_list = wector_layer.shp_df
        ...

    @staticmethod
    def get_list_check(uri, file_extension):
        path = uri
        file_list = []
        for r, d, f in os.walk(path):
            for file in f:
                if file_extension in file:
                    fileName = os.path.join(r, file)
                    file_list.append(fileName)
        return file_list

    def create_instances_of_pc(self):
        lista = [Lasek(x) for x in self.list_of_pc]
        # list_of_box = [x.bbox_poligon_offset for x in lista]
        df_of_box = pd.DataFrame((zip(self.list_of_pc, lista)), columns=['path_to_las', 'Lasek']).set_index('path_to_las')
        return df_of_box


    def pkt_is_in_cloud(self, pkt):
        test = self.df_of_lasek['Lasek'].apply((lambda m: m.point_is_in(pkt)))
        x = self.df_of_lasek[test== True]
        return x

    def map_by_elements_with_progress_bar(self, my_function: 'function', list_of_object:[], *args, **kwargs):
        """PChandler.save_segments lub PChandler.save_for_uv"""

        pbar = tqdm(total=len(list_of_object), leave=False)
        pbar.set_description(f'Wycinanie elementow z chmury ')
        for las in list_of_object:
            try:
                my_function(self, las, kwargs)
            except ValueError:
                pass
            except ToSmallCloud:
                pass
            except Exception as e:
                tqdm.write(f'{e} - nie udalo sie przetworzyc chmury {las}')
            finally:
                pbar.update(1)
        del pbar

    def save_segments(self, pc, save_las=True, save_txt=False):
        """Zapisywanie chmury do las lub txt"""
        try:
            one_cloud = Lasek(pc)
        except Exception as e:
            raise e

        mask = self.lista_points.apply(lambda x: one_cloud.point_is_in(x), axis=1)
        point_included = self.lista_points[mask]
        if len(point_included) == 0:
            raise ValueError('zero points in pc')

        np_lasek = one_cloud.read_np_array()

        for pkt in point_included.iterrows():
            try:
                clip = Lasek_clip(np_lasek, pkt)
                las = clip.las_np
                if save_las:
                    clip.save_pc_las(las, os.path.join(self.path_save, str(pkt[0]) + '.las'))
                if save_txt:
                    clip.save_pc_txt(las, os.path.join(self.path_save, str(pkt[0]) + '.csv'))
            except ToSmallCloud as e:
                pass
            except Exception as e:
                print(e)

    def save_for_uv(self, las, save_txt=True):
        """Zapis na potrzeby skryptu do przeliczania"""
        try:
            one_cloud = Lasek(las)
        except Exception as e:
            raise e

        point_name = one_cloud.name[:-4]
        try:
            pkt = self.lista_points.loc[int(point_name)]
        except ValueError:
            pkt = self.lista_points.loc[point_name]
        except Exception as e:
            raise e('Nie ma takiego rekordu')

        np_lasek = one_cloud.read_np_array()
        try:
            clip = Lasek_clip(np_lasek, pkt, True)
            las = clip.clip_to_box()
            if save_txt:
                clip.save_pc_txt(las, os.path.join(self.path_save, str(point_name) + '.csv'))
        except Exception as e:
            print(e)


def main():
    start = datetime.now()
    time = start.time().strftime('%H%M%S')

    home_data = os.getcwd()

    shp_path = os.path.join(home_data, 'data', 'object', '7_Z1_N2_LK8_5m.txt')
    path_to_las = os.path.join(home_data, 'data', 'point_cloud')
    path_to_save = os.path.join(home_data, 'out')

    file_to_save = os.path.join(path_to_save, time)
    os.makedirs(path_to_save, exist_ok=1)

    print(f'Start: {start}')

    # Fragment do wyciecia chmurki okrąg
    pc_handler = PChandler(path_to_las, path_to_save, shp_path)
    pc_handler.map_by_elements_with_progress_bar(PChandler.save_segments, pc_handler.list_of_pc, save_las=True, save_txt=False)

    # # Fragment do wyciecia chmurki poligon i zapisanie do dalszych przeliczeń UV
    pc_handler = PChandler(path_to_las, path_to_save, shp_path)
    pc_handler.map_by_elements_with_progress_bar(PChandler.save_for_uv, pc_handler.get_list_check(path_to_save, '.las'), save_txt=True)

    print('\n', datetime.now() - start, ', ', psutil.virtual_memory())
    print('end')

    # input('puszczac main_XYZ?')

    # import modulu do przeliczenia punktów z chmury na zdjecie
    module_dir = os.path.join(os.path.dirname(home_data), 'XYZtoUV')

    sys.path.append(module_dir)
    import main_XYZ_to_uv

    path_to_proj = os.path.join(module_dir, 'data', 'pix_proj', '7_Z1_N2_PKP_L8_part1.p4d')
    photos_path = os.path.join(module_dir, 'data', 'photos')
    name = 'labels.csv'

    path_save_photo = path_to_save

    result_tst_file = os.path.join(path_save_photo, name)

    # przeliczenie z chmury na na zdjęcie
    main_XYZ_to_uv.test(path_to_proj, photos_path, path_save_photo, result_tst_file)

    # prezentacja wynikow
    main_XYZ_to_uv.show_mi(path_save_photo, result_tst_file)


if __name__ == '__main__':
    main()