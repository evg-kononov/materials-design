expr = '''
toFEM[x_,num_]:=(
    Get["NDSolve`FEM`"];
    Get["MeshTools`"];
    
    rrb=Image3D[x];
    uniquenumber=num;
    directoryname=FileNameJoin[{"samples",ToString[num]}];
    x0=y0=z0=0;
    xA=yA=zA=2;
    rrbm=RegionResize[ImageMesh[rrb,Method->"DualMarchingCubes"],{{x0,xA},{y0,yA},{z0,zA}}];
    rrn=ColorNegate[rrb];
    rrnm=RegionResize[ImageMesh[rrn,Method->"DualMarchingCubes"],{{x0,xA},{y0,yA},{z0,zA}}];
    
    matrixmesh=NDSolve`FEM`ToElementMesh[rrbm,"RegionHoles"->RandomPoint[rrnm,5]];
    assembly=MeshTools`AddMeshMarkers[matrixmesh,"MeshElementsMarker"->2];
    twophaseflag=False;
    
    elements1=Pick[assembly[[2,1,1]],assembly[[2,1,2]],1];
    elements1=Take[#,4]&/@elements1; (* Taking only first 4 coordinates of each element *) 
    elem1number=Length@elements1;
    elements2=Pick[assembly[[2,1,1]],assembly[[2,1,2]],2];
    elements2=Take[#,4]&/@elements2; (* Taking only first 4 coordinates of each element *) 
    elem2number=Length@elements2;
    elemtotalnumber=elem1number+elem2number;
    
    (* List of nodes *)
    NodesFromElements1=Sort[DeleteDuplicates[Flatten[elements1]]]; (* Selects nodes that are used in LINEAR elements1 *)
    NodesFromElements2=Sort[DeleteDuplicates[Flatten[elements2]]]; (* Selects nodes that are used in LINEAR elements2 *)
    NodesFromAllElements=Union[NodesFromElements1,NodesFromElements2]; (* Selects nodes that are used for all LINEAR elements *)
    coordinates=assembly["Coordinates"]; (* Selects all coordinates *)
    (*coordinates//Short*)
    coordnumber=Length[coordinates];
    voxelflag=False;
    nodestext=Map[ToString,Flatten/@Table[{i,NumberForm[#,10,ExponentFunction->(Null&)]&/@coordinates[[i]]},{i,NodesFromAllElements}],{2}]; (* Selects LINEAR coordinates and translates to string   *)
    (nodestext=MapAt[StringInsert[#,",",-1]&,nodestext,{All,{1,2,3}}])//Grid;
    nodestext=StringJoin/@nodestext;
    ncoordinates=Map[ToString,Flatten/@Table[{NumberForm[#,10,ExponentFunction->(Null&)]&/@coordinates[[i]]},{i,NodesFromAllElements}],{2}]; 
    ncoordinates=Map[ToExpression,ncoordinates,{2}]; 
    If[twophaseflag==True,
    elements1text=Map[ToString,Flatten/@Table[{i,elements1[[i]]},{i,elem1number}],{2}];
    (elements1text=MapAt[StringInsert[#,",",-1]&,elements1text,{All,{1,2,3,4}}])//Grid;
    elements1text=StringJoin/@elements1text;];
    elements2text=Map[ToString,Flatten/@Table[{elem1number+i,elements2[[i]]},{i,elem2number}],{2}];
    (elements2text=MapAt[StringInsert[#,",",-1]&,elements2text,{All,{1,2,3,4}}])//Grid;
    elements2text=StringJoin/@elements2text;
    If[twophaseflag==True,
    set1elementstext=ToString/@{1,elem1number,1};
    set1elementstext=MapAt[StringInsert[#,",",-1]&,set1elementstext,{{1},{2}}];
    set1elementstext={StringJoin[set1elementstext]};];
    set2elementstext=ToString/@{elem1number+1,elem1number+elem2number,1};
    set2elementstext=MapAt[StringInsert[#,",",-1]&,set2elementstext,{{1},{2}}];
    set2elementstext={StringJoin[set2elementstext]};
    ClearAll[MatrixX0Text,MatrixXAText,MatrixZ0Text,MatrixZAText,MatrixFaceText];
    MatrixY0Text=MatrixYAText=MatrixXAText=MatrixX0Text=MatrixZ0Text=MatrixZAText=MatrixFaceText={"**"};
    tol=10^-1;(* Nodes selection tolerance*)
    y0select=Select[ncoordinates,(N@Round[#[[2]],tol]==N@Round[y0,tol])&]; 
    y0nodes=Flatten[Position[ncoordinates,{_,Alternatives@@(DeleteDuplicates[y0select[[All,2]]]),_}]];
    yAselect=Select[ncoordinates,(N@Round[#[[2]],tol]==N@Round[yA,tol])&];
    yAnodes=Flatten[Position[ncoordinates,{_,Alternatives@@(DeleteDuplicates[yAselect[[All,2]]]),_}]];
    y0nodes=Intersection[y0nodes,NodesFromAllElements];
    yAnodes=Intersection[yAnodes,NodesFromAllElements];
    MatrixY0Text=Map[ToString,Flatten/@Partition[y0nodes,UpTo[16]],{2}];
    (MatrixY0Text=MapAt[StringInsert[#,",",-1]&,MatrixY0Text,{All}])//Grid;
    MatrixY0Text=StringJoin/@MatrixY0Text;
    MatrixYAText=Map[ToString,Flatten/@Partition[yAnodes,UpTo[16]],{2}];
    (MatrixYAText=MapAt[StringInsert[#,",",-1]&,MatrixYAText,{All}])//Grid;
    MatrixYAText=StringJoin/@MatrixYAText;
    z0select=Select[ncoordinates,(N@Round[#[[3]],tol]==N@Round[z0,tol])&]; 
    z0nodes=Flatten[Position[coordinates,{_,_,Alternatives@@(DeleteDuplicates[z0select[[All,3]]])}]];
    zAselect=Select[ncoordinates,(N@Round[#[[3]],tol]==N@Round[zA,tol])&];
    zAnodes=Flatten[Position[coordinates,{_,_,Alternatives@@(DeleteDuplicates[zAselect[[All,3]]])}]];
    z0nodes=Intersection[z0nodes,NodesFromAllElements];
    zAnodes=Intersection[zAnodes,NodesFromAllElements];
    MatrixZ0Text=Map[ToString,Flatten/@Partition[z0nodes,UpTo[16]],{2}];
    (MatrixZ0Text=MapAt[StringInsert[#,",",-1]&,MatrixZ0Text,{All}])//Grid;
    MatrixZ0Text=StringJoin/@MatrixZ0Text;
    MatrixZAText=Map[ToString,Flatten/@Partition[zAnodes,UpTo[16]],{2}];
    (MatrixZAText=MapAt[StringInsert[#,",",-1]&,MatrixZAText,{All}])//Grid;
    MatrixZAText=StringJoin/@MatrixZAText;
    MatrixX0elemsText=MatrixXAelemsText=MatrixY0elemsText=MatrixYAelemsText=MatrixZ0elemsText=MatrixZAelemsText={"**"};
    
    (* Abaqus INP file - Static Analysis *)
    (** INP TEXT **)
    HeadingText={"*Heading","**(c) Mikhail Tashkinov","**","**PARTS","**","*Part, name=RVE","*NODE, NSET=AllNodes"};
    ElementText=If[voxelflag==False,{"*Element, type=C3D4"},{"*Element, type=C3D8"}(*elements for voxel model*)];
    
    Set1Text=Join[{"*Elset, elset=Inclusions, generate"},set1elementstext];
    Set2Text=Join[{"*Elset, elset=Matrix, generate"},set2elementstext];
    Set2X0Text=Join[{"*Elset, elset=Elem_MatrixX0"},MatrixX0elemsText];
    Set2XAText=Join[{"*Elset, elset=Elem_MatrixXA"},MatrixXAelemsText];
    Set2Y0Text=Join[{"*Elset, elset=Elem_MatrixY0"},MatrixY0elemsText];
    Set2YAText=Join[{"*Elset, elset=Elem_MatrixYA"},MatrixYAelemsText];
    Set2Z0Text=Join[{"*Elset, elset=Elem_MatrixZ0"},MatrixZ0elemsText];
    Set2ZAText=Join[{"*Elset, elset=Elem_MatrixZA"},MatrixZAelemsText];
    
    Set2XTopText=Join[{"*Nset, nset=Matrix_XA"},MatrixXAText];
    Set2XBottomText=Join[{"*Nset, nset=Matrix_X0"},MatrixX0Text];
    Set2YTopText=Join[{"*Nset, nset=Matrix_YA"},MatrixYAText];
    Set2YBottomText=Join[{"*Nset, nset=Matrix_Y0"},MatrixY0Text];
    Set2ZTopText=Join[{"*Nset, nset=Matrix_ZA"},MatrixZAText];
    Set2ZBottomText=Join[{"*Nset, nset=Matrix_Z0"},MatrixZ0Text];
    
    
    SectionText={"*Orientation,name=Ori-1","1.,0.,0.,0.,1.,0.","1,0.",
    "**Section:Inclusions",
    "*Solid Section,elset=INCLUSIONS,orientation=Ori-1,controls=EC-1,material=HIPS",",",
    "**Section:Matrix",
    "*Solid Section,elset=MATRIX,orientation=Ori-1,controls=EC-1,material=PLA_05_180",","};
    
    SectionTextPorous={"*Orientation,name=Ori-1","1.,0.,0.,0.,1.,0.","1,0.",
    "**Section:Matrix","*Solid Section,elset=MATRIX,orientation=Ori-1,controls=EC-1,material=PLA_05_180",","};
    EndPartText={"*End Part"};
    
    
    AssemblyText={"**",
    "**","**ASSEMBLY",
    "**","*Assembly,name=Assembly",
    "**","*Instance,name=RVE-1,part=RVE","*End Instance",
    "**",
    "*Node","1,"<>ToString@N[(xA-x0)/2]<>","<>ToString@N[(yA-y0)/2]<>","<>ToString@N[z0-(zA-z0)/10],
    "*Node","2,"<>ToString@N[(xA-x0)/2]<>","<>ToString@N[(yA-y0)/2]<>","<>ToString@N[zA+(zA-z0)/10],
    "*Node","3,"<>ToString@N[(xA-x0)/2]<>","<>ToString@N[y0-(yA-y0)/10]<>","<>ToString@N[(zA-z0)/2],
    "*Node","4,"<>ToString@N[(xA-x0)/2]<>","<>ToString@N[yA+(yA-y0)/10]<>","<>ToString@N[(zA-z0)/2],
    "*Node","5,"<>ToString@N[x0-(xA-x0)/10]<>","<>ToString@N[(yA-y0)/2]<>","<>ToString@N[(zA-z0)/2],
    "*Node","6,"<>ToString@N[x0+(xA-x0)/10]<>","<>ToString@N[(yA-y0)/2]<>","<>ToString@N[(zA-z0)/2],
    "*Nset,nset=RF_Z0",
    "1,",
    "*Nset,nset=RF_ZA",
    "2,",
    "*Nset,nset=RF_Y0",
    "3,",
    "*Nset,nset=RF_YA",
    "4,",
    "*Nset,nset=RF_X0",
    "5,",
    "*Nset,nset=RF_XA",
    "6,",
    "*Surface, type=NODE, name=RVE-1_MATRIX_ZA_CNS_, internal","RVE-1.MATRIX_ZA, 1.","*Surface, type=NODE, name=RVE-1_MATRIX_Z0_CNS_, internal","RVE-1.MATRIX_Z0, 1.",
    "**Constraint:Constraint-Z0",
    "*MPC",
    "TIE,RVE-1.MATRIX_Z0,RF_Z0",
    "**Constraint:Constraint-ZA",
    "*MPC",
    "TIE,RVE-1.MATRIX_ZA,RF_ZA","*End Assembly"};
    
    (* For standard *)
    MiscTextPorous={"**",
    "** MATERIALS",
    "**",
    "*Material, name=ABS_04","*Density"," 1.05e-09,","*Elastic","5235.75, 0.392",
    "*Material, name=ABS_08","*Density"," 1.05e-09,","*Elastic","5841.26, 0.392",
    "*Material, name=Carbon","*Density"," 1.72e-09,","*Elastic","220000., 0.15",
    "*Material, name=Glass","*Density"," 2.61e-09,","*Elastic","70000., 0.32",
    "*Material, name=Basalt","*Density"," 2.65e-09,","*Elastic","89000., 0.23",
    "*Material, name=HIPS","*Density"," 1.3e-09,","*Elastic","2000., 0.35",
    "*Material, name=PLA_08_180","*Density"," 1.25e-09,","*Elastic","2580., 0.36",
    "*Material, name=PLA_05_180","*Density"," 1.25e-09,","*Elastic","2620., 0.36",
    (*"** ",
    "** INTERACTION PROPERTIES",
    "** ",
    "*Surface Interaction, name=GenContact","*Friction","0.,","*Surface Behavior, pressure-overclosure=HARD",*)
    "** ",
    "** BOUNDARY CONDITIONS",
    "** ",
    "** Name: Fix_Bottom Type: Displacement/Rotation","*Boundary","RF_Z0, 1, 1","RF_Z0, 2, 2","RF_Z0, 3, 3","RF_Z0, 4, 4","RF_Z0, 5, 5","RF_Z0, 6, 6",
    "** Name: Fix_Top Type: Displacement/Rotation","*Boundary","RF_ZA, 1, 1","RF_ZA, 2, 2","RF_ZA, 4, 4","RF_ZA, 5, 5","RF_ZA, 6, 6",
    "**",
    "**STEP:Step-1",
    "**",
    "*Step, name=Step-1, nlgeom=NO, inc=10000",
    "*Static",
    "1, 1., 1e-25, 1",
    "**",
    "**LOADS",
    "**",
    "**Name:Load-1 Type:Concentrated force",
    "*Cload",
    "RF_ZA,3,500.",
    "**",
    "**OUTPUT REQUESTS",
    "**","*Restart,write,frequency=0",
    "**",
    "**FIELD OUTPUT:F-Output-1",
    "**",
    "*Output,field","*Node Output","RF,U","*Element Output,directions=YES","EVOL,IVOL,E,S",
    "**",
    "**HISTORY OUTPUT:H-Output-1",
    "**",
    "*Output,history,variable=PRESELECT",
    (*"**",
    "*FILE OUTPUT, number interval=20",
    "*NODE FILE, nset=PLATE-2.REF_POINT",
    "RF, U",*)
    "*End Step"};
    
    
    exporttextporous=Join[HeadingText,nodestext,ElementText,elements2text,Set2Text,Set2X0Text,Set2XAText,Set2Y0Text,Set2YAText,Set2Z0Text,Set2ZAText,Set2YBottomText,Set2YTopText,Set2XTopText,Set2XBottomText,Set2ZTopText,Set2ZBottomText,SectionTextPorous,EndPartText,EndPartText,AssemblyText,MiscTextPorous];
    
    If[twophaseflag,exporttext2phase=Join[HeadingText,nodestext,ElementText,elements1text,elements2text,Set1Text,Set2Text,Set2X0Text,Set2XAText,Set2Y0Text,Set2YAText,Set2Z0Text,Set2ZAText,Set2YBottomText,Set2YTopText,Set2XTopText,Set2XBottomText,Set2ZTopText,Set2ZBottomText,SectionText,EndPartText,EndPartText,AssemblyText,MiscTextPorous];];
    
    
    CreateDirectory[directoryname];
    SetDirectory[directoryname];
    
    
    porousname=StringJoin["structure_"<>ToString[uniquenumber],"_porous_ZLoad.inp"];
    Export[porousname,exporttextporous,"Text"];
    
    
    If[twophaseflag,twophasename=StringJoin["structure_"<>ToString[uniquenumber],"_2phase_ZLoad.inp"];
    Export[twophasename,exporttext2phase,"Text"]];
    )
'''
