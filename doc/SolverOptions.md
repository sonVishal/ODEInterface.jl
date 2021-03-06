[ This file was auto-generated from the module's documentation included in the doc-strings. Use julia's help system to get these informations in a nicer output format. ]

# dopri5

```
function dopri5(rhs::Function, t0::Real, T::Real,
                x0::Vector, opt::AbstractOptionsODE)
     -> (t,x,retcode,stats)
```

`retcode` can have the following values:

```
  1: computation successful
  2: computation. successful, but interrupted by output function
 -1: input is not consistent
 -2: larger OPT_MAXSTEPS is needed
 -3: step size becomes too small
 -4: problem is probably stiff (interrupted)
```

main call for using Fortran-dopri5 solver. In `opt` the following options are used:

<table>
<tr><th><pre>  Option
</pre></th>
<th><pre> Description
</pre></th>
<th><pre> Default
</pre></th>
</tr>
<tr><td><pre> RTOL     &#38;
 ATOL
</pre></td>
<td><pre> relative and absolute error tolerances
 both scalars or both vectors with the
 length of length&#40;x0&#41;
 error&#40;x&#8342;&#41; &#8804; OPT&#95;RTOL&#8342;&#8901;&#124;x&#8342;&#124;&#43;OPT&#95;ATOL&#8342;
</pre></td>
<td><pre>    1e&#45;3
    1e&#45;6
</pre></td>
</tr>
<tr><td><pre> OUTPUTFCN
</pre></td>
<td><pre> output function
 see help&#95;outputfcn
</pre></td>
<td><pre> nothing
</pre></td>
</tr>
<tr><td><pre> OUTPUTMODE
</pre></td>
<td><pre> OUTPUTFCN&#95;NEVER&#58;
   dont&#39;t call OPT&#95;OUTPUTFCN
 OUTPUTFCN&#95;WODENSE
   call OPT&#95;OUTPUTFCN&#44; but without
   possibility for dense output
 OUTPUTFCN&#95;DENSE
   call OPT&#95;OUTPUTFCN with support for
   dense output
</pre></td>
<td><pre>   NEVER
</pre></td>
</tr>
<tr><td><pre> MAXSTEPS
</pre></td>
<td><pre> maximal number of allowed steps
 OPT&#95;MAXSTEPS &#62; 0
</pre></td>
<td><pre>  100000
</pre></td>
</tr>
<tr><td><pre> STEST
</pre></td>
<td><pre> stiffness test
 done after every step number k&#42;OPT&#95;STEST
 OPT&#95;STEST &#60; 0 for turning test off
 OPT&#95;STEST &#8800; 0
</pre></td>
<td><pre>    1000
</pre></td>
</tr>
<tr><td><pre> EPS
</pre></td>
<td><pre> the rounding unit
 1e&#45;35 &#60; OPT&#95;EPS &#60; 1&#46;0
</pre></td>
<td><pre> 2&#46;3e&#45;16
</pre></td>
</tr>
<tr><td><pre> RHO
</pre></td>
<td><pre> safety factor in step size predcition
 1e&#45;4  &#60; OPT&#95;RHO &#60; 1&#46;0
</pre></td>
<td><pre>     0&#46;9
</pre></td>
</tr>
<tr><td><pre> SSMINSEL   &#38;
 SSMAXSEL
</pre></td>
<td><pre> parameters for step size selection
 The new step size is chosen subject to
 the restriction
 OPT&#95;SSMINSEL &#8804; hnew&#47;hold &#8804; OPT&#95;SSMAXSEL
 OPT&#95;SSMINSEL&#44; OPT&#95;SSMAXSEL &#62; 0
</pre></td>
<td><pre>     0&#46;2
    10&#46;0
</pre></td>
</tr>
<tr><td><pre> SSBETA
</pre></td>
<td><pre> &#946; for stabilized step size control
 OPT&#95;SSBETA &#8804; 0&#46;2
 if OPT&#95;SSBETA &#60; 0 then OPT&#95;SSBETA &#61; 0
</pre></td>
<td><pre>    0&#46;04
</pre></td>
</tr>
<tr><td><pre> MAXSS
</pre></td>
<td><pre> maximal step size
 OPT&#95;MAXSS &#8800; 0
</pre></td>
<td><pre>  T &#45; t0
</pre></td>
</tr>
<tr><td><pre> INITIALSS
</pre></td>
<td><pre> initial step size
 if OPT&#95;INITIALSS &#61;&#61; 0 then a initial
 guess is computed
</pre></td>
<td><pre>     0&#46;0
</pre></td>
</tr>
</table>


# dop853

```
function dop853(rhs::Function, t0::Real, T::Real,
                x0::Vector, opt::AbstractOptionsODE)
     -> (t,x,retcode,stats)
```

`retcode` can have the following values:

```
  1: computation successful
  2: computation. successful, but interrupted by output function
 -1: input is not consistent
 -2: larger OPT_MAXSTEPS is needed
 -3: step size becomes too small
 -4: problem is probably stiff (interrupted)
```

main call for using Fortran-dopri5 solver. In `opt` the following options are used:

<table>
<tr><th><pre>  Option OPT&#95;&#8230;
</pre></th>
<th><pre> Description
</pre></th>
<th><pre> Default
</pre></th>
</tr>
<tr><td><pre> RTOL     &#38;
 ATOL
</pre></td>
<td><pre> relative and absolute error tolerances
 both scalars or both vectors with the
 length of length&#40;x0&#41;
 error&#40;x&#8342;&#41; &#8804; OPT&#95;RTOL&#8342;&#8901;&#124;x&#8342;&#124;&#43;OPT&#95;ATOL&#8342;
</pre></td>
<td><pre>    1e&#45;3
    1e&#45;6
</pre></td>
</tr>
<tr><td><pre> OUTPUTFCN
</pre></td>
<td><pre> output function
 see help&#95;outputfcn
</pre></td>
<td><pre> nothing
</pre></td>
</tr>
<tr><td><pre> OUTPUTMODE
</pre></td>
<td><pre> OUTPUTFCN&#95;NEVER&#58;
   dont&#39;t call OPT&#95;OUTPUTFCN
 OUTPUTFCN&#95;WODENSE
   call OPT&#95;OUTPUTFCN&#44; but without
   possibility for dense output
 OUTPUTFCN&#95;DENSE
   call OPT&#95;OUTPUTFCN with support for
   dense output
</pre></td>
<td><pre>   NEVER
</pre></td>
</tr>
<tr><td><pre> MAXSTEPS
</pre></td>
<td><pre> maximal number of allowed steps
 OPT&#95;MAXSTEPS &#62; 0
</pre></td>
<td><pre>  100000
</pre></td>
</tr>
<tr><td><pre> STEST
</pre></td>
<td><pre> stiffness test
 done after every step number k&#42;OPT&#95;STEST
 OPT&#95;STEST &#60; 0 for turning test off
 OPT&#95;STEST &#8800; 0
</pre></td>
<td><pre>    1000
</pre></td>
</tr>
<tr><td><pre> EPS
</pre></td>
<td><pre> the rounding unit
 1e&#45;35 &#60; OPT&#95;EPS &#60; 1&#46;0
</pre></td>
<td><pre> 2&#46;3e&#45;16
</pre></td>
</tr>
<tr><td><pre> RHO
</pre></td>
<td><pre> safety factor in step size predcition
 1e&#45;4  &#60; OPT&#95;RHO &#60; 1&#46;0
</pre></td>
<td><pre>     0&#46;9
</pre></td>
</tr>
<tr><td><pre> SSMINSEL   &#38;
 SSMAXSEL
</pre></td>
<td><pre> parameters for step size selection
 The new step size is chosen subject to
 the restriction
 OPT&#95;SSMINSEL &#8804; hnew&#47;hold &#8804; OPT&#95;SSMAXSEL
 OPT&#95;SSMINSEL&#44; OPT&#95;SSMAXSEL &#62; 0
</pre></td>
<td><pre>   0&#46;333
     6&#46;0
</pre></td>
</tr>
<tr><td><pre> SSBETA
</pre></td>
<td><pre> &#946; for stabilized step size control
 OPT&#95;SSBETA &#8804; 0&#46;2
 if OPT&#95;SSBETA &#60; 0 then OPT&#95;SSBETA &#61; 0
</pre></td>
<td><pre>     0&#46;0
</pre></td>
</tr>
<tr><td><pre> MAXSS
</pre></td>
<td><pre> maximal step size
 OPT&#95;MAXSS &#8800; 0
</pre></td>
<td><pre>  T &#45; t0
</pre></td>
</tr>
<tr><td><pre> INITIALSS
</pre></td>
<td><pre> initial step size
 if OPT&#95;INITIALSS &#61;&#61; 0 then a initial
 guess is computed
</pre></td>
<td><pre>     0&#46;0
</pre></td>
</tr>
</table>


# odex

```
 function odex(rhs::Function, t0::Real, T::Real, 
               x0::Vector, opt::AbstractOptionsODE)
     -> (t,x,retcode,stats)
```

`retcode` can have the following values:

```
  1: computation successful
  2: computation. successful, but interrupted by output function
 -1: error
```

main call for using Fortran-odex solver. In `opt` the following options are used:

<table>
<tr><th><pre>  Option OPT&#95;&#8230;
</pre></th>
<th><pre> Description
</pre></th>
<th><pre> Default
</pre></th>
</tr>
<tr><td><pre> RTOL         &#38;
 ATOL
</pre></td>
<td><pre> relative and absolute error tolerances
 both scalars or both vectors with the
 length of length&#40;x0&#41;
 error&#40;x&#8342;&#41; &#8804; OPT&#95;RTOL&#8342;&#8901;&#124;x&#8342;&#124;&#43;OPT&#95;ATOL&#8342;
</pre></td>
<td><pre>    1e&#45;3
    1e&#45;6
</pre></td>
</tr>
<tr><td><pre> OUTPUTFCN
</pre></td>
<td><pre> output function
 see help&#95;outputfcn
</pre></td>
<td><pre> nothing
</pre></td>
</tr>
<tr><td><pre> OUTPUTMODE
</pre></td>
<td><pre> OUTPUTFCN&#95;NEVER&#58;
   dont&#39;t call OPT&#95;OUTPUTFCN
 OUTPUTFCN&#95;WODENSE
   call OPT&#95;OUTPUTFCN&#44; but without
   possibility for dense output
 OUTPUTFCN&#95;DENSE
   call OPT&#95;OUTPUTFCN with support for
   dense output
</pre></td>
<td><pre>   NEVER
</pre></td>
</tr>
<tr><td><pre> MAXSTEPS
</pre></td>
<td><pre> maximal number of allowed steps
 OPT&#95;MAXSTEPS &#62; 0
</pre></td>
<td><pre>   10000
</pre></td>
</tr>
<tr><td><pre> EPS
</pre></td>
<td><pre> the rounding unit
 1e&#45;35 &#60; OPT&#95;EPS &#60; 1&#46;0
</pre></td>
<td><pre> 2&#46;3e&#45;16
</pre></td>
</tr>
<tr><td><pre> MAXSS
</pre></td>
<td><pre> maximal step size
 OPT&#95;MAXSS &#8800; 0
</pre></td>
<td><pre>  T &#45; t0
</pre></td>
</tr>
<tr><td><pre> INITIALSS
</pre></td>
<td><pre> initial step size guess
</pre></td>
<td><pre>    1e&#45;4
</pre></td>
</tr>
<tr><td><pre> MAXEXCOLUMN
</pre></td>
<td><pre> the maximum number of columns in
 the extrapolation table
 OPT&#95;MAXEXCOLUMN &#8805; 3
</pre></td>
<td><pre>       9
</pre></td>
</tr>
<tr><td><pre> STEPSIZESEQUENCE
</pre></td>
<td><pre> switch for the step size sequence
 1&#58; 2&#44; 4&#44;  6&#44;  8&#44; 10&#44; 12&#44; 14&#44; 16&#44; &#8230;
 2&#58; 2&#44; 4&#44;  8&#44; 12&#44; 16&#44; 20&#44; 24&#44; 28&#44; &#8230;
 3&#58; 2&#44; 4&#44;  6&#44;  8&#44; 12&#44; 16&#44; 24&#44; 32&#44; &#8230;
 4&#58; 2&#44; 6&#44; 10&#44; 14&#44; 18&#44; 22&#44; 26&#44; 30&#44; &#8230;
 5&#58; 4&#44; 8&#44; 12&#44; 16&#44; 20&#44; 24&#44; 28&#44; 32&#44; &#8230;
 1 &#8804; OPT&#95;STEPSIZESEQUENCE &#8804; 5
</pre></td>
<td><pre>       4
 if
 OUTPUT&#45;
 MODE &#61;&#61;
 DENSE&#59;
 other&#45;
 wise  1
</pre></td>
</tr>
<tr><td><pre> MAXSTABCHECKS
</pre></td>
<td><pre> how many times is the stability check
 activated at most in one line of the
 extrapolation table
</pre></td>
<td><pre>       1
</pre></td>
</tr>
<tr><td><pre> MAXSTABCHECKLINE
</pre></td>
<td><pre> stability check is only activated in
 the lines 1 to MAXMAXSTABCHECKLINE of
 the extrapolation table
</pre></td>
<td><pre>       1
</pre></td>
</tr>
<tr><td><pre> DENSEOUTPUTWOEE
</pre></td>
<td><pre> boolean flag&#58; suppress error estimator
 in dense output
 true is only possible&#44; if
      OUTPUTMODE &#61;&#61; DENSE
</pre></td>
<td><pre>   false
</pre></td>
</tr>
<tr><td><pre> INTERPOLDEGREE
</pre></td>
<td><pre> determines the degree of interpolation
 formula&#58;
 &#956; &#61; 2&#42;&#954; &#45; INTERPOLDEGREE &#43; 1
</pre></td>
<td><pre>       4
</pre></td>
</tr>
<tr><td><pre> SSREDUCTION
</pre></td>
<td><pre> step size is reduced by factor if the
 stability check is negative
 OPT&#95;EPS &#60; OPT&#95;SSREDUCTION &#60; 1
</pre></td>
<td><pre>     0&#46;5
</pre></td>
</tr>
<tr><td><pre> SSSELECTPAR1 &#38;
 SSSELECTPAR2
</pre></td>
<td><pre> parameters for step size selection
 the new step size for the k&#45;th diagonal
 entry is chosen subject to
 FMIN&#47;SSSELECTPAR2 &#8804; hnew&#8342;&#47;hold &#8804; 1&#47;FMIN
 with FMIN &#61; SSSELECTPAR1&#94;&#40;1&#47;&#40;2&#42;k&#45;1&#41;&#41;
</pre></td>
<td><pre>    0&#46;02
    4&#46;00
</pre></td>
</tr>
<tr><td><pre> ORDERDECFRAC &#38;
 ORDERINCFRAC
</pre></td>
<td><pre> parameters for the order selection
 decrease order if
         W&#40;k&#45;1&#41; &#8804;   W&#40;k&#41;&#42;ORDERDECFRAC
 increase order if
         W&#40;k&#41;   &#8804; W&#40;k&#45;1&#41;&#42;ORDERINCFRAC
</pre></td>
<td><pre>     0&#46;8
     0&#46;9
</pre></td>
</tr>
<tr><td><pre> OPT&#95;RHO      &#38;
 OPT&#95;RHO2
</pre></td>
<td><pre> safety factors for step control algorithm
 hnew&#61;h&#42;RHO&#42;&#40;RHO2&#42;TOL&#47;ERR&#41;&#94;&#40;1&#47;&#40;k&#45;1&#41; &#41;
</pre></td>
<td><pre>    0&#46;94
    0&#46;65
</pre></td>
</tr>
</table>


# seulex

```
 function seulex(rhs::Function, t0::Real, T::Real,
                 x0::Vector, opt::AbstractOptionsODE)
     -> (t,x,retcode,stats)
```

`retcode` can have the following values:

```
  1: computation successful
  2: computation. successful, but interrupted by output function
 -1: computation unsuccessful
```

main call for using Fortran seulex solver.

This solver support problems with special structure, see `help_specialstructure`.

In `opt` the following options are used:

<table>
<tr><th><pre>  Option OPT&#95;&#8230;
</pre></th>
<th><pre> Description
</pre></th>
<th><pre> Default
</pre></th>
</tr>
<tr><td><pre> RHSAUTONOMOUS
</pre></td>
<td><pre> Flag&#44; if right&#45;hand side is autonomous&#46;
</pre></td>
<td><pre>   false
</pre></td>
</tr>
<tr><td><pre> M1 &#38; M2
</pre></td>
<td><pre> parameter for special structure&#44; see
 above
 M1&#44; M2 &#8805; 0
 M1 &#43;M2 &#8804; length&#40;x0&#41;
 &#40;M1&#61;&#61;M2&#61;&#61;0&#41; &#124;&#124; &#40;M1&#8800;0&#8800;M2&#41;
 M1 &#37; M2 &#61;&#61; 0 or M1&#61;&#61;0
</pre></td>
<td><pre>       0
      M1
</pre></td>
</tr>
<tr><td><pre> RTOL         &#38;
 ATOL
</pre></td>
<td><pre> relative and absolute error tolerances
 both scalars or both vectors with the
 length of length&#40;x0&#41;
 error&#40;x&#8342;&#41; &#8804; OPT&#95;RTOL&#8342;&#8901;&#124;x&#8342;&#124;&#43;OPT&#95;ATOL&#8342;
</pre></td>
<td><pre>    1e&#45;3
    1e&#45;6
</pre></td>
</tr>
<tr><td><pre> OUTPUTFCN
</pre></td>
<td><pre> output function
 see help&#95;outputfcn
</pre></td>
<td><pre> nothing
</pre></td>
</tr>
<tr><td><pre> OUTPUTMODE
</pre></td>
<td><pre> OUTPUTFCN&#95;NEVER&#58;
   dont&#39;t call OPT&#95;OUTPUTFCN
 OUTPUTFCN&#95;WODENSE
   call OPT&#95;OUTPUTFCN&#44; but without
   possibility for dense output
 OUTPUTFCN&#95;DENSE
   call OPT&#95;OUTPUTFCN with support for
   dense output
</pre></td>
<td><pre>   NEVER
</pre></td>
</tr>
<tr><td><pre> LAMBDADENSE
</pre></td>
<td><pre> parameter &#955; of dense output
 OPT&#95;LAMBDADENSE &#8712; &#123;0&#44;1&#125;
</pre></td>
<td><pre>       0
</pre></td>
</tr>
<tr><td><pre> EPS
</pre></td>
<td><pre> the rounding unit
 0 &#60; OPT&#95;EPS &#60; 1&#46;0
</pre></td>
<td><pre>   1e&#45;16
</pre></td>
</tr>
<tr><td><pre> TRANSJTOH
</pre></td>
<td><pre> The solver transforms the jacobian
 matrix to Hessenberg form&#46;
 This option is not supported if the
 system is &#34;implicit&#34; &#40;i&#46;e&#46; a mass matrix
 is given&#41; or if jacobian is banded&#46;
</pre></td>
<td><pre>   false
</pre></td>
</tr>
<tr><td><pre> MAXSTEPS
</pre></td>
<td><pre> maximal number of allowed steps
 OPT&#95;MAXSTEPS &#62; 0
</pre></td>
<td><pre>  100000
</pre></td>
</tr>
<tr><td><pre> MAXSS
</pre></td>
<td><pre> maximal step size
 OPT&#95;MAXSS &#8800; 0
</pre></td>
<td><pre>  T &#45; t0
</pre></td>
</tr>
<tr><td><pre> INITIALSS
</pre></td>
<td><pre> initial step size guess
</pre></td>
<td><pre>    1e&#45;6
</pre></td>
</tr>
<tr><td><pre> MAXEXCOLUMN
</pre></td>
<td><pre> the maximum number of columns in
 the extrapolation table
 OPT&#95;MAXEXCOLUMN &#8805; 3
</pre></td>
<td><pre>      12
</pre></td>
</tr>
<tr><td><pre> STEPSIZESEQUENCE
</pre></td>
<td><pre> switch for the step size sequence
 1&#58; 1&#44; 2&#44; 3&#44; 6&#44; 8&#44; 12&#44; 16&#44; 24&#44; 32&#44; 48&#44; &#8230;
 2&#58; 2&#44; 3&#44; 4&#44; 6&#44; 8&#44; 12&#44; 16&#44; 24&#44; 32&#44; 48&#44; &#8230;
 3&#58; 1&#44; 2&#44; 3&#44; 4&#44; 5&#44;  6&#44;  7&#44;  8&#44;  9&#44; 10&#44; &#8230;
 4&#58; 2&#44; 3&#44; 4&#44; 5&#44; 6&#44;  7&#44;  8&#44;  9&#44; 10&#44; 11&#44; &#8230;
 1 &#8804; OPT&#95;STEPSIZESEQUENCE &#8804; 4
</pre></td>
<td><pre>       2
</pre></td>
</tr>
<tr><td><pre> SSSELECTPAR1 &#38;
 SSSELECTPAR2
</pre></td>
<td><pre> parameters for step size selection
 the new step size for the k&#45;th diagonal
 entry is chosen subject to
 FMIN&#47;SSSELECTPAR2 &#8804; hnew&#8342;&#47;hold &#8804; 1&#47;FMIN
 with FMIN &#61; SSSELECTPAR1&#94;&#40;1&#47;&#40;k&#45;1&#41;&#41;
</pre></td>
<td><pre>     0&#46;1
     4&#46;0
</pre></td>
</tr>
<tr><td><pre> ORDERDECFRAC &#38;
 ORDERINCFRAC
</pre></td>
<td><pre> parameters for the order selection
 decrease order if
         W&#40;k&#45;1&#41; &#8804;   W&#40;k&#41;&#42;ORDERDECFRAC
 increase order if
         W&#40;k&#41;   &#8804; W&#40;k&#45;1&#41;&#42;ORDERINCFRAC
</pre></td>
<td><pre>     0&#46;7
     0&#46;9
</pre></td>
</tr>
<tr><td><pre> JACRECOMPFACTOR
</pre></td>
<td><pre> decides whether the jacobian should be
 recomputed&#46;
 small &#40;&#8776; 0&#46;001&#41;&#58; recompute often
 large &#40;&#8776; 0&#46;1&#41;&#58; recompute rarely
 i&#46;e&#46; this number represents how costly
 Jacobia evaluations are&#46;
 OPT&#95;JACRECOMPFACTOR &#8800; 0
</pre></td>
<td><pre> min&#40;
   1e&#45;4&#44;
 RTOL&#91;1&#93;&#41;
</pre></td>
</tr>
<tr><td><pre> OPT&#95;RHO      &#38;
 OPT&#95;RHO2
</pre></td>
<td><pre> safety factors for step control algorithm
 hnew&#61;h&#42;RHO&#42;&#40;RHO2&#42;TOL&#47;ERR&#41;&#94;&#40;1&#47;&#40;k&#45;1&#41; &#41;
</pre></td>
<td><pre>    0&#46;93
    0&#46;80
</pre></td>
</tr>
<tr><td><pre> MASSMATRIX
</pre></td>
<td><pre> the mass matrix of the problem&#46; If not
 given &#40;nothing&#41; then the identiy matrix
 is used&#46;
 The size has to be &#40;d&#45;M1&#41;&#215;&#40;d&#45;M1&#41;&#46;
 It can be an full matrix or a banded
 matrix &#40;BandedMatrix&#41;&#46;
</pre></td>
<td><pre> nothing
</pre></td>
</tr>
<tr><td><pre> JACOBIMATRIX
</pre></td>
<td><pre> A function providing the Jacobian for
 &#8706;f&#47;&#8706;x or nothing&#46; For nothing the solver
 uses finite differences to calculate the
 Jacobian&#46;
 The function has to be of the form&#58;
   function &#40;t&#44;x&#44;J&#41; &#45;&#62; nothing       &#40;A&#41;
 or for M1&#62;0 &#38; JACOBIBANDSTRUCT &#8800; nothing
   function &#40;t&#44;x&#44;J1&#44;&#8230;&#44;JK&#41; &#45;&#62; nothing &#40;B&#41;
 with K &#61; 1&#43;M1&#47;M2 and &#40;M1&#43;M2&#61;&#61;d&#41;
 see help&#95;specialstructure
</pre></td>
<td><pre> nothing
</pre></td>
</tr>
<tr><td><pre> JACOBIBANDSTRUCT
</pre></td>
<td><pre> A tuple &#40;l&#44;u&#41; describing the banded
 structure of the Jacobian or nothing if
 the Jacobian is full&#46;
 see help&#95;specialstructure
</pre></td>
<td><pre> nothing
</pre></td>
</tr>
<tr><td><pre> WORKFORRHS
 WORKFORJAC
 WORKFORDEC
 WORKFORSOL
</pre></td>
<td><pre> estimated works &#40;complexity&#41; for a call
 to
 WORKFORRHS&#58; right&#45;hand side f
 WORKFORJAC&#58; JACOBIMATRIX
 WORKFORDEC&#58; LU&#45;decomposition
 WORKFORSOL&#58; Forward&#45; and Backward subst&#46;
</pre></td>
<td><pre>     1&#46;0
     5&#46;0
     1&#46;0
     1&#46;0
</pre></td>
</tr>
</table>


# radau and radau5

```
 function radau(rhs::Function, t0::Real, T::Real,
                 x0::Vector, opt::AbstractOptionsODE)
     -> (t,x,retcode,stats)
 
 function radau5(rhs::Function, t0::Real, T::Real,
                 x0::Vector, opt::AbstractOptionsODE)
     -> (t,x,retcode,stats)
```

`retcode` can have the following values:

```
  1: computation successful
  2: computation. successful, but interrupted by output function
 -1: input is not consistent
 -2: larger OPT_MAXSTEPS is needed
 -3: step size becomes too small
 -4: matrix is repeatedly singular
```

main call for using Fortran radau or radau5 solver.

This solver support problems with special structure, see `help_specialstructure`.

Remark: Because radau and radau5 are collocation methods, there is no difference  in the computational costs for OUTPUTFCN_WODENSE and OUTPUTFCN_DENSE.

In `opt` the following options are used:

<table>
<tr><th><pre>  Option OPT&#95;&#8230;
</pre></th>
<th><pre> Description
</pre></th>
<th><pre> Default
</pre></th>
</tr>
<tr><td><pre> M1 &#38; M2
</pre></td>
<td><pre> parameter for special structure&#44; see
 above
 M1&#44; M2 &#8805; 0
 M1 &#43;M2 &#8804; length&#40;x0&#41;
 &#40;M1&#61;&#61;M2&#61;&#61;0&#41; &#124;&#124; &#40;M1&#8800;0&#8800;M2&#41;
 M1 &#37; M2 &#61;&#61; 0 or M1&#61;&#61;0
</pre></td>
<td><pre>       0
      M1
</pre></td>
</tr>
<tr><td><pre> RTOL         &#38;
 ATOL
</pre></td>
<td><pre> relative and absolute error tolerances
 both scalars or both vectors with the
 length of length&#40;x0&#41;
 error&#40;x&#8342;&#41; &#8804; OPT&#95;RTOL&#8342;&#8901;&#124;x&#8342;&#124;&#43;OPT&#95;ATOL&#8342;
</pre></td>
<td><pre>    1e&#45;3
    1e&#45;6
</pre></td>
</tr>
<tr><td><pre> OUTPUTFCN
</pre></td>
<td><pre> output function
 see help&#95;outputfcn
</pre></td>
<td><pre> nothing
</pre></td>
</tr>
<tr><td><pre> OUTPUTMODE
</pre></td>
<td><pre> OUTPUTFCN&#95;NEVER&#58;
   dont&#39;t call OPT&#95;OUTPUTFCN
 OUTPUTFCN&#95;WODENSE
   call OPT&#95;OUTPUTFCN&#44; but without
   possibility for dense output
 OUTPUTFCN&#95;DENSE
   call OPT&#95;OUTPUTFCN with support for
   dense output
</pre></td>
<td><pre>   NEVER
</pre></td>
</tr>
<tr><td><pre> EPS
</pre></td>
<td><pre> the rounding unit
 1e&#45;19 &#60; OPT&#95;EPS &#60; 1&#46;0
</pre></td>
<td><pre>   1e&#45;16
</pre></td>
</tr>
<tr><td><pre> TRANSJTOH
</pre></td>
<td><pre> The solver transforms the jacobian
 matrix to Hessenberg form&#46;
 This option is not supported if the
 system is &#34;implicit&#34; &#40;i&#46;e&#46; a mass matrix
 is given&#41; or if jacobian is banded&#46;
</pre></td>
<td><pre>   false
</pre></td>
</tr>
<tr><td><pre> MAXNEWTONITER
</pre></td>
<td><pre> maximum number of Newton iterations for
 the solution of the implicit system in
 each step&#46;
 for radau&#58; MAXNEWTONITER &#43; &#40;NS&#45;3&#41;&#42;2&#46;5
   where NS is number of current stages
 for radau5&#58;     OPT&#95;MAXNEWTONITER &#62; 0
 for radau&#58; 50 &#62; OPT&#95;MAXNEWTONITER &#62; 0
</pre></td>
<td><pre>       7
</pre></td>
</tr>
<tr><td><pre> NEWTONSTARTZERO
</pre></td>
<td><pre> if &#96;false&#96;&#44; the extrapolated collocation
 solution is taken as starting vector for
 Newton&#39;s method&#46; If &#96;true&#96; zero starting
 values are used&#46; The latter is
 recommended if Newton&#39;s method has
 difficulties with convergence&#46;
</pre></td>
<td><pre>   false
</pre></td>
</tr>
<tr><td><pre> NEWTONSTOPCRIT
</pre></td>
<td><pre> only for radau5&#58;
 Stopping criterion for Newton&#39;s method&#46;
 Smaller values make the code slower&#44; but
 safer&#46;
 Default&#58;
  max&#40;10&#42;OPT&#95;EPS&#47;OPT&#95;RTOL&#91;1&#93;&#44;
       min&#40;0&#46;03&#44;sqrt&#40;OPT&#95;RTOL&#91;1&#93;&#41;&#41;&#41;
 OPT&#95;NEWTONSTOPCRIT &#62; OPT&#95;EPS&#47;OPT&#95;RTOL&#91;1&#93;
</pre></td>
<td><pre> see
   left
</pre></td>
</tr>
<tr><td><pre> DIMOFIND1VAR  &#38;
 DIMOFIND2VAR  &#38;
 DIMOFIND3VAR
</pre></td>
<td><pre> For differential&#45;algebraic systems of
 index &#62; 1&#46; The right&#45;hand side should be
 written such that the index 1&#44;2&#44;3
 variables appear in this order&#46;
 DIMOFINDzVAR&#58; number of index z vars&#46;
 &#8721; DIMOFINDzVAR &#61;&#61; length&#40;x0&#41;
</pre></td>
<td><pre> len&#40;x0&#41;
       0
       0
</pre></td>
</tr>
<tr><td><pre> MAXSTEPS
</pre></td>
<td><pre> maximal number of allowed steps
 OPT&#95;MAXSTEPS &#62; 0
</pre></td>
<td><pre>  100000
</pre></td>
</tr>
<tr><td><pre> MAXSS
</pre></td>
<td><pre> maximal step size
 OPT&#95;MAXSS &#8800; 0
</pre></td>
<td><pre>  T &#45; t0
</pre></td>
</tr>
<tr><td><pre> INITIALSS
</pre></td>
<td><pre> initial step size guess
</pre></td>
<td><pre>    1e&#45;6
</pre></td>
</tr>
<tr><td><pre> MINSTAGES     &#38;
 MAXSTAGES
</pre></td>
<td><pre> only for radau&#58;
 minimal and maximal number of stages&#46;
 The order is given by&#58; 2&#8901;stages&#45;1
 MINSTAGES&#44;MAXSTAGES &#8712; &#40;1&#44;3&#44;5&#44;7&#41;
 MINSTAGES &#8804; MAXSTAGES
</pre></td>
<td><pre>       3
       7
</pre></td>
</tr>
<tr><td><pre> INITSTAGES
</pre></td>
<td><pre> only for radau&#58;
 number of stages to start with&#46;
 MINSTAGES &#8804; INITSTAGES &#8804; MAXSTAGES
</pre></td>
<td><pre>MINSTAGES
</pre></td>
</tr>
<tr><td><pre> STEPSIZESTRATEGY
</pre></td>
<td><pre> Switch for step size strategy
   1&#58; mod&#46; predictive controller
      &#40;Gustafsson&#41;
   2&#58; classical step size control
</pre></td>
<td><pre>       1
</pre></td>
</tr>
<tr><td><pre> OPT&#95;RHO
</pre></td>
<td><pre> safety factor for step control algorithm
 0&#46;001 &#60; OPT&#95;RHO &#60; 1&#46;0
</pre></td>
<td><pre>     0&#46;9
</pre></td>
</tr>
<tr><td><pre> JACRECOMPFACTOR
</pre></td>
<td><pre> decides whether the jacobian should be
 recomputed&#46;
 &#60;0&#58; recompute after every accepted step
 small &#40;&#8776; 0&#46;001&#41;&#58; recompute often
 large &#40;&#8776; 0&#46;1&#41;&#58; recompute rarely
 i&#46;e&#46; this number represents how costly
 Jacobia evaluations are&#46;
 OPT&#95;JACRECOMPFACTOR &#8800; 0
</pre></td>
<td><pre>   0&#46;001
</pre></td>
</tr>
<tr><td><pre> FREEZESSLEFT  &#38;
 FREEZESSRIGHT
</pre></td>
<td><pre> Step size freezing&#58; If
 FREEZESSLEFT &#60; hnew&#47;hold &#60; FREEZESSRIGHT
 then the step size is not changed&#46; This
 saves&#44; together with a large
 JACRECOMPFACTOR&#44; LU&#45;decompositions and
 computing time for large systems&#46;
 small systems&#58;
    FREEZESSLEFT  &#8776; 1&#46;0
    FREEZESSRIGHT &#8776; 1&#46;2
 large full systems&#58;
    FREEZESSLEFT  &#8776; 0&#46;99
    FREEZESSRIGHT &#8776; 2&#46;0
 OPT&#95;FREEZESSLEFT  &#8804; 1&#46;0
 OPT&#95;FREEZESSRIGHT &#8805; 1&#46;0
</pre></td>
<td><pre>     1&#46;0
     1&#46;2
</pre></td>
</tr>
<tr><td><pre> SSMINSEL   &#38;
 SSMAXSEL
</pre></td>
<td><pre> parameters for step size selection
 The new step size is chosen subject to
 the restriction
 OPT&#95;SSMINSEL &#8804; hnew&#47;hold &#8804; OPT&#95;SSMAXSEL
 OPT&#95;SSMINSEL &#8804; 1&#44; OPT&#95;SSMAXSEL &#8805; 1
</pre></td>
<td><pre>     0&#46;2
     8&#46;0
</pre></td>
</tr>
<tr><td><pre> MASSMATRIX
</pre></td>
<td><pre> the mass matrix of the problem&#46; If not
 given &#40;nothing&#41; then the identiy matrix
 is used&#46;
 The size has to be &#40;d&#45;M1&#41;&#215;&#40;d&#45;M1&#41;&#46;
 It can be an full matrix or a banded
 matrix &#40;BandedMatrix&#41;&#46;
</pre></td>
<td><pre> nothing
</pre></td>
</tr>
<tr><td><pre> JACOBIMATRIX
</pre></td>
<td><pre> A function providing the Jacobian for
 &#8706;f&#47;&#8706;x or nothing&#46; For nothing the solver
 uses finite differences to calculate the
 Jacobian&#46;
 The function has to be of the form&#58;
   function &#40;t&#44;x&#44;J&#41; &#45;&#62; nothing       &#40;A&#41;
 or for M1&#62;0 &#38; JACOBIBANDSTRUCT &#8800; nothing
   function &#40;t&#44;x&#44;J1&#44;&#8230;&#44;JK&#41; &#45;&#62; nothing &#40;B&#41;
 with K &#61; 1&#43;M1&#47;M2 and &#40;M1&#43;M2&#61;&#61;d&#41;
 see help&#95;specialstructure
</pre></td>
<td><pre> nothing
</pre></td>
</tr>
<tr><td><pre> JACOBIBANDSTRUCT
</pre></td>
<td><pre> A tuple &#40;l&#44;u&#41; describing the banded
 structure of the Jacobian or nothing if
 the Jacobian is full&#46;
 see help&#95;specialstructure
</pre></td>
<td><pre> nothing
</pre></td>
</tr>
<tr><td><pre> ORDERDECFACTOR &#38;
 ORDERINCFACTOR
</pre></td>
<td><pre> only for radau&#58;
 Order is decreased&#44; if the contractivity
 factor is smaller than ORDERDECFACTOR&#46;
 Order is increased&#44; if the contractivity
 factor is larger than ORDERINCFACTOR&#46;
 ORDERDECFACTOR &#62; ORDERINCFACTOR &#62; 0
</pre></td>
<td><pre>     0&#46;8
   0&#46;002
</pre></td>
</tr>
<tr><td><pre> ORDERDECSTEPFAC1
 ORDERDECSTEPFAC2
</pre></td>
<td><pre> only for radau&#58;
 the order is only decreased if the
 stepsize ratio satisfies
  ORDERDECSTEPFAC2 &#8804; hnew&#47;hold &#8804;
               ORDERDECSTEPFAC1
 0 &#60; ORDERDECSTEPFAC2 &#60; ORDERDECSTEPFAC1
</pre></td>
<td><pre>     1&#46;2
     0&#46;8
</pre></td>
</tr>
</table>


# bvpsol

```
 function bvpsol(rhs::Function, bc::Function,
   t::Vector, x::Matrix, odesolver, opt::AbstractOptionsODE)
     -> (t,x,retcode,stats)
```

The `bc` has to be a function of the following form:

```
 function bc(xa,xb,r) -> nothing
```

It has to calculate the residual for the boundary conditions and save them in `r`.

`t` is a Vector with all the multiple-shooting nodes.

`x` gives the initial guess for all multiple-shooting nodes. Hence `size(x,2)==length(t)`.

`odesolver`: Either `nothing`: then the internal solver of `bvpsol` is used. Or `odesolver` is a ode-solver (like `dopri5`, `dop853`, `seulex`,  etc.).

`retcode` can have the following values:

```
  >0: computation successful: number of iterations
  -1:        Iteration stops at stationary point for OPT_SOLMETHOD==0
             Gaussian elimination failed due to singular 
             Jacobian for OPT_SOLMETHOD==1
  -2: Iteration stops after OPT_MAXSTEPS 
  -3: Integrator failed to complete the trajectory
  -4: Gauss Newton method failed to converge
  -5: Given initial values inconsistent with separable linear bc
  -6:        Iterative refinement faild to converge for OPT_SOLMETHOD==0
             Termination since multiple shooting condition or
             condition of Jacobian is too bad for OPT_SOLMETHOD==1
  -7: wrong EPS (should not happen; checked by ODEInterface module)
  -8: Condensing algorithm for linear block system fails, try
      OPT_SOLMETHOD==1
  -9: Sparse linear solver failed
 -10: Real or integer work-space exhausted
 -11: Rank reduction failed - resulting rank is zero
```

In `opt` the following options are used:

<table>
<tr><th><pre>  Option OPT&#95;&#8230;
</pre></th>
<th><pre> Description
</pre></th>
<th><pre> Default
</pre></th>
</tr>
<tr><td><pre> RTOL
</pre></td>
<td><pre> relative accuracy for soltuion
</pre></td>
<td><pre>    1e&#45;6
</pre></td>
</tr>
<tr><td><pre> MAXSTEPS
</pre></td>
<td><pre> maximum permitted number of iteration
 steps
</pre></td>
<td><pre>      40
</pre></td>
</tr>
<tr><td><pre> BVPCLASS
</pre></td>
<td><pre> boundary value problem classification&#58;
 0&#58; linear
 1&#58; nonlinear with good initial data
 2&#58; highly nonlinear &#38; bad initial data
 3&#58; highly nonlinear &#38; bad initial data &#38;
    initial rank reduction to separable
    linear boundary conditions
</pre></td>
<td><pre>       2
</pre></td>
</tr>
<tr><td><pre> SOLMETHOD
</pre></td>
<td><pre> switch for solution method
 0&#58; use local linear solver with
    condensing algorithm
 1&#58; use global sparse linear solver
</pre></td>
<td><pre>       0
</pre></td>
</tr>
<tr><td><pre> IVPOPT
</pre></td>
<td><pre> An OptionsODE&#45;object with the options
 for the solver of the initial value
 problem&#46;
 In this OptionsODE&#45;object bvpsol changes
 OPT&#95;MAXSS&#44; OPT&#95;INITIALSS&#44; OPT&#95;RTOL
 to give the IVP&#45;solver solution hints&#46;
</pre></td>
<td><pre> empty
 options
</pre></td>
</tr>
<tr><td><pre> RHS&#95;CALLMODE
</pre></td>
<td><pre> see help&#95;callsolvers&#40;&#41;
</pre></td>
<td><pre></pre></td>
</tr>
</table>


