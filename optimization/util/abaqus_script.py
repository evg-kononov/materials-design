# -*- coding: mbcs -*-
# Do not delete the following import lines
import time

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

import sys
import os

# --------------------------------------------------------------------
#   Name of Modules
# --------------------------------------------------------------------

NameOfPart = "RVE"
NameOfInstance = NameOfPart + "-1"

# --------------------------------------------------------------------
#   Assembly Creation
# --------------------------------------------------------------------

L = 2.0

X_Min = 0.0
Y_Min = 0.0
Z_Min = 0.0
X_Max = L
Y_Max = L
Z_Max = L


def calc_young(start_idx, load_path, save_path):
    """
    load_path - file path where the inp file paths are stored
    save_path - file path where to save the joung modules
    """

    with open(load_path) as f:
        inp_paths = f.read().splitlines()
    for idx, inp_path in enumerate(inp_paths[start_idx:]):
        print(idx, inp_path)
        with open("sys_exit.txt", "w+") as f:
            f.write(str(idx + start_idx))
        print " "
        print " "
        print " "
        job_name = "Jobs"
        inp_path = inp_path[:-4]  # [:-4] - обрезает ".inp"
        file_name = inp_path.split("\\")[-1]
        print file_name + ".inp"

        mdb.ModelFromInputFile(name=file_name, inputFileName=inp_path + ".inp")

        """
        #--------------------------------------------------------------------
        #   Sets of nodes 
        #--------------------------------------------------------------------

        #   Торцы по направлениям (X-)(Y-)(Z-), индексы поверхностей face1Elements, face4Elements, face5Elements
        nodes1 = mdb.models[inp_path].parts[NameOfPart].nodes.getByBoundingBox(X_Min, Y_Min, Z_Min, X_Min, Y_Max, Z_Max)
        mdb.models[inp_path].parts[NameOfPart].Set(nodes=nodes1, name="Side X_A")
        nodes1 = mdb.models[inp_path].parts[NameOfPart].nodes.getByBoundingBox(X_Min, Y_Min, Z_Min, X_Max, Y_Min, Z_Max)
        mdb.models[inp_path].parts[NameOfPart].Set(nodes=nodes1, name="Side Y_A")
        nodes1 = mdb.models[inp_path].parts[NameOfPart].nodes.getByBoundingBox(X_Min, Y_Min, Z_Min, X_Max, Y_Max, Z_Min)
        mdb.models[inp_path].parts[NameOfPart].Set(nodes=nodes1, name="Side Z_A")

        #   Торец по направлению (X+), индекс поверхностей face2Elements
        nodes1 = mdb.models[inp_path].parts[NameOfPart].nodes.getByBoundingBox(X_Max, Y_Min, Z_Min, X_Max, Y_Max, Z_Max)
        mdb.models[inp_path].parts[NameOfPart].Set(nodes=nodes1, name="Side X_B") #точки на торце X_Max

        #   Торец по направлению (Y+), индекс поверхностей face6Elements
        nodes1 = mdb.models[inp_path].parts[NameOfPart].nodes.getByBoundingBox(X_Min, Y_Max, Z_Min, X_Max, Y_Max, Z_Max)
        mdb.models[inp_path].parts[NameOfPart].Set(nodes=nodes1, name="Side Y_B") #точки на торце Y_Max

        #   Торец по направлению (Z+), индекс поверхностей face3Elements
        nodes1 = mdb.models[inp_path].parts[NameOfPart].nodes.getByBoundingBox(X_Min, Y_Min, Z_Max, X_Max, Y_Max, Z_Max)
        mdb.models[inp_path].parts[NameOfPart].Set(nodes=nodes1, name="Side Z_B") #точки на торце Z_Max
        """

        a = mdb.models[file_name].rootAssembly
        session.viewports["Viewport: 1"].setValues(displayedObject=a)

        if a.getMassProperties()["volume"] is None:
            continue
        else:
            VF = a.getMassProperties()["volume"] / L**3

        mdb.Job(name=job_name, model=file_name, description="",
                type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None,
                memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
                explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
                modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine="",
                scratch="", resultsFormat=ODB, numThreadsPerMpiProcess=1,
                multiprocessingMode=DEFAULT, numCpus=12, numDomains=12, numGPUs=0)
        mdb.jobs[job_name].submit(consistencyChecking=OFF)
        mdb.jobs[job_name].waitForCompletion()

        # Открываем посчитавшийся .odb
        try:
            odb = session.openOdb(name=job_name + ".odb")
        except Exception as ex:
            sys.exit()

        session.viewports["Viewport: 1"].setValues(displayedObject=odb)

        xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
                                                                                         "RF", NODAL,
                                                                                         ((COMPONENT, "RF3"),)),),
                                            nodeSets=("RF_Z0",))

        RF = abs(sum(xyList)[-1][-1])
        print("RF =", RF)

        xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
                                                                                         "U", NODAL,
                                                                                         ((COMPONENT, "U3"),)),),
                                            nodeSets=("RF_ZA",))
        U = abs(sum(xyList)[-1][-1])
        print("U =", U)

        E = (RF * L) / (U * L ** 2)
        print("E =", E)
        print("VF =", VF)

        with open(save_path, "a") as f:
            f.write(inp_path + "," + str(E) + "," + str(VF) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        calc_young(int(sys.argv[-3]), sys.argv[-2], sys.argv[-1])
        with open("sys_exit.txt", "w+") as f:
            f.write("end")
    else:
        print("Add the args!")