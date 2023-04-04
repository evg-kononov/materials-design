# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__

import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior

#--------------------------------------------------------------------
#   Name of Modules
#--------------------------------------------------------------------

NameOfPart = 'RVE'
NameOfInstance = NameOfPart + '-1'

#--------------------------------------------------------------------
#   Assembly Creation
#--------------------------------------------------------------------

L = 2.0

X_Min = 0.0
Y_Min = 0.0
Z_Min = 0.0
X_Max = L
Y_Max = L
Z_Max = L



# Путь к рабочей директории
work_directory = 'D:/abaqusWorking/KseniyaSyp/'
# Открываем файл с названиями inp'ов (some_name.inp)
with open('file_list.txt') as f:
    file_names = f.read().splitlines()

for idx, file_name in enumerate(file_names):
   
    print ' '
    print ' '
    print ' '
    job_name = 'Jobs'
    file_name = file_name[:-4] # [:-4] - обрезает '.inp'
    print(file_name)
    
    mdb.ModelFromInputFile(name=file_name, 
        inputFileName=work_directory + file_name + '.inp')
       
    """
    #--------------------------------------------------------------------
    #   Sets of nodes 
    #--------------------------------------------------------------------

    #   Торцы по направлениям (X-)(Y-)(Z-), индексы поверхностей face1Elements, face4Elements, face5Elements
    nodes1 = mdb.models[file_name].parts[NameOfPart].nodes.getByBoundingBox(X_Min, Y_Min, Z_Min, X_Min, Y_Max, Z_Max)
    mdb.models[file_name].parts[NameOfPart].Set(nodes=nodes1, name='Side X_A')
    nodes1 = mdb.models[file_name].parts[NameOfPart].nodes.getByBoundingBox(X_Min, Y_Min, Z_Min, X_Max, Y_Min, Z_Max)
    mdb.models[file_name].parts[NameOfPart].Set(nodes=nodes1, name='Side Y_A')
    nodes1 = mdb.models[file_name].parts[NameOfPart].nodes.getByBoundingBox(X_Min, Y_Min, Z_Min, X_Max, Y_Max, Z_Min)
    mdb.models[file_name].parts[NameOfPart].Set(nodes=nodes1, name='Side Z_A')

    #   Торец по направлению (X+), индекс поверхностей face2Elements
    nodes1 = mdb.models[file_name].parts[NameOfPart].nodes.getByBoundingBox(X_Max, Y_Min, Z_Min, X_Max, Y_Max, Z_Max)
    mdb.models[file_name].parts[NameOfPart].Set(nodes=nodes1, name='Side X_B') #точки на торце X_Max

    #   Торец по направлению (Y+), индекс поверхностей face6Elements
    nodes1 = mdb.models[file_name].parts[NameOfPart].nodes.getByBoundingBox(X_Min, Y_Max, Z_Min, X_Max, Y_Max, Z_Max)
    mdb.models[file_name].parts[NameOfPart].Set(nodes=nodes1, name='Side Y_B') #точки на торце Y_Max

    #   Торец по направлению (Z+), индекс поверхностей face3Elements
    nodes1 = mdb.models[file_name].parts[NameOfPart].nodes.getByBoundingBox(X_Min, Y_Min, Z_Max, X_Max, Y_Max, Z_Max)
    mdb.models[file_name].parts[NameOfPart].Set(nodes=nodes1, name='Side Z_B') #точки на торце Z_Max
    """
    
    a = mdb.models[file_name].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    mdb.Job(name=job_name, model=file_name, description='', 
        type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, 
        memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1, 
        multiprocessingMode=DEFAULT, numCpus=12, numDomains=12, numGPUs=0)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()
  
    
    # Открываем посчитавшийся .odb
    odb = session.openOdb(name=work_directory + job_name + '.odb')
    session.viewports['Viewport: 1'].setValues(displayedObject=odb)
    
    xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
        'RF', NODAL, ((COMPONENT, 'RF3'), )), ), nodeSets=("RF_Z0", ))
    RF = abs(sum(xyList)[-1][-1])
    print('RF =', RF)
    
    xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
        'U', NODAL, ((COMPONENT, 'U3'), )), ), nodeSets=("RF_ZA", ))
    U = abs(sum(xyList)[-1][-1])
    print('U =', U)
    
    E = (RF * L) / (U * L**2)
    print('E =', E)
    
    

