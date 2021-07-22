import os
import sys

#GASOIL
path = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\PruebaErroresGasoil\\"
pathsalida = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\SALIDAS\\"


#GASOIL #MISMO FILTRADO QUE PRUEBASERRORESGASOIL FILTRÓ NINGUNO
#path = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\GasoilPruebasGG\\"
#pathsalida = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\SalidaPruebasGG\\"

#GASOIL ESCALADO X
#path = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\PruebaErroresGasoilESCALADOX\\"
#pathsalida = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\SalidaPruebaErroresGasoilESCALADOX\\"


#GASOIL ESCALADO X / Y
#path = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\PruebaErroresGasoilESCALADOXY\\"
#pathsalida = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\SalidaPruebaErroresGasoilESCALADOXY\\"


#SUPER
#path = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\SUPERESCALADOX\\"
#pathsalida = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\SALIDASUPERESCALADOX\\"

#PREMIUM N12
#path = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\Premium1\\N12\\"
#pathsalida = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\SalidaPremium1\\N12\\"

#PREMIUM N24
#path = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\Premium1\\N24"
#pathsalida = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\SalidaPremium1\\N24"



#path = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\PruebaErroresGasoil\\"
#pathsalida = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\ResultadosExp\\SalidaPruebaErroresGasoil\\"
dicFiltrado = { "fil_promedio_total": 9, # el promedio total no puede ser mayor de 6.5
				"fil_promedio_anual": 7, # el promedio anual no puede ser mayor de 7
				"tolerancia_pa": 4, # el promedio anual no puede pasar de 7 mas de 3 veces
				"fil_pa_max": 7, # el promedio anual NO puede pasar 8 
				"fil_error_mensual": 4, # el error mensual no puede ser mayor de 8
				"tolerancia_mensual": 4, # e; error mensual se puede pasar solo 4 veces de 8
				"cant_ejecuciones_exc_tol_men": 5, # cant de ejecuciones que se aceptan tengan mas de 4 meses mas de 8
				"fil_error_mensual_max": 25 # maximo error de mes aceptado
			  }

fpt = dicFiltrado["fil_promedio_total"]  #filtro promedio total

fpa = dicFiltrado["fil_promedio_anual"]  #filtro promedio anual
tpa = dicFiltrado["tolerancia_pa"]     #tolerancia anual, se puede pasar del limite una cierta cantidad en los promedios anuales
fpam = dicFiltrado["fil_pa_max"]

fm  = dicFiltrado["fil_error_mensual"] #filtro error mensual
tm  = dicFiltrado["tolerancia_mensual"]    #tolerancia mensual, se puede pasar del limite una cierta cantidad en los errores mensuales / cant meses fuera de rango
tam = dicFiltrado["cant_ejecuciones_exc_tol_men"]    #tolerancia anios mensual / tolerancia de anios con una cantidad de meses q se fueron de rango
fmm = dicFiltrado["fil_error_mensual_max"]

total = len(os.listdir(path))
for nombre in os.listdir(pathsalida):
    if os.path.isfile(pathsalida + nombre):
        os.remove(pathsalida + nombre)
cantidad = 0
for nombre in os.listdir(path): 
	if os.path.isfile(path + nombre):

		archivo = open(path + nombre,"r") #abrimos el archivo en modo lectura
		linea=archivo.readline()
		try:
			resultados = eval(linea) #evalua el string que se le pasa como un objeto, y el readline lee la linea en la que estas parado en el archivo

			if (resultados["ept"] > fpt) and (fpt > 0):
				continue
		except:
			continue
		cepa = 0	
		for pa in resultados["epa"]:
			if (pa > fpam) and (fpam > 0) :
				cepa = tpa	+ 1
				break
			if (pa > fpa) and (fpa > 0) :
				cepa += 1

		if cepa > tpa:
			continue

		cam = 0
		for anio in resultados["em"]:		
			cm = 0
			for em in anio:
				if( em > fmm) and (fmm > 0):
					cam = tam + 1
					break
				if (em > fm) and (fm > 0):
					cm += 1
			if cm > tm:
				cam += 1	
			if cam > tam:
				break

		if cam > tam:	
			continue



		archivosalida = open(pathsalida + nombre,"w")


		for linea in archivo:
			archivosalida.write(linea)

		cantidad += 1
		archivosalida.close()
		archivo.close()

print("Total: {}\nFiltrado: {}".format(total, cantidad))










