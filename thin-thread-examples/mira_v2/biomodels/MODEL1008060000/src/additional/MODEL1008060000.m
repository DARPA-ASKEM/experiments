
#
# This file is automatically generated with 
# the System Biology Format Converter (http://sbfc.sourceforge.net/)
# from an SBML file.
#
# The conversion system has the following limitations:
#  - You may have to re order some reactions and Assignment Rules definition
#  - Delays are not taken into account
#  - You should change the lsode parameters (start, end, steps) to get better results
#

#
# The following line is there to be sure that Octave think that this file 
# is a script and not function file
#
1;

#
# Model name = Munz2009 - Zombi Impulsive Killing
#
# is http://identifiers.org/biomodels.db/MODEL1008060000
# isDescribedBy http://identifiers.org/isbn/ISBN:1607413477
#
function z=pow(x,y),z=x^y;endfunction
function z=root(x,y),z=y^(1/x);endfunction
function z = piecewise(varargin)
	numArgs = nargin;
	result = 0;
	foundResult = 0;
	for k=1:2: numArgs-1
		if varargin{k+1} == 1
			result = varargin{k};
			foundResult = 1;
			break;
		endif
	end
	if foundResult == 0
		result = varargin{numArgs};
	endif
z = result;
endfunction

function xdot=f(x,t)
# Compartment: id = env, name = environment, constant
	compartment_env=1.0;
# Parameter:   id =  N, name = starting Population
	global_par_N=500.0;
# Parameter:   id =  p, name = birth rate
# Parameter:   id =  delta, name = delta
	global_par_delta=1.0E-4;
# Parameter:   id =  beta, name = beta
	global_par_beta=0.0055;
# Parameter:   id =  zeta, name = zeta
	global_par_zeta=0.09;
# Parameter:   id =  alpha, name = alpha
	global_par_alpha=0.0075;
# Parameter:   id =  n, name = number of kills
	global_par_n=0.0;
# Parameter:   id =  k, name = kill ratio
	global_par_k=0.25;
# Parameter:   id =  tau, name = kill intervall
	global_par_tau=2.5;
# assignmentRule: variable = p
	global_par_p=x(1)*global_par_delta;
# Reaction: id = birth


	reaction_birth=global_par_p;
# Reaction: id = death


	reaction_death=global_par_delta*x(1);
# Reaction: id = infection


	reaction_infection=global_par_beta*x(1)*x(2);
# Reaction: id = resurrection


	reaction_resurrection=global_par_zeta*x(3);
# Reaction: id = destruction


	reaction_destruction=global_par_alpha*x(1)*x(2);

#Event: id=zombie_eradication
	event_zombie_eradication=(time >= (global_par_n+1)*global_par_tau) and (global_par_k*(global_par_n+1) <= 1);

	if(event_zombie_eradication) 
		x(2)=x(2)*(1-global_par_k*(global_par_n+1));
		global_par_n=global_par_n+1;
	endif
	xdot=zeros(3,1);
	# Species:   id = S, name = Susceptible, affected by kineticLaw

	xdot(1) = ( 1.0 * reaction_birth) + (-1.0 * reaction_death) + (-1.0 * reaction_infection);

	# Species:   id = Z, name = Zombie, affected by kineticLaw

	xdot(2) = ( 1.0 * reaction_infection) + ( 1.0 * reaction_resurrection) + (-1.0 * reaction_destruction);

	# Species:   id = R, name = Removed, affected by kineticLaw

	xdot(3) = ( 1.0 * reaction_death) + (-1.0 * reaction_resurrection) + ( 1.0 * reaction_destruction);

endfunction

#Initial conditions vector
x0=zeros(3,1);
x0(1) = 0;
x0(2) = 1.0;
x0(3) = 0.0;


#Creating linespace
t=linspace(0,90,100);

#Solving equations
x=lsode("f",x0,t);

#ploting the results
plot(t,x);
