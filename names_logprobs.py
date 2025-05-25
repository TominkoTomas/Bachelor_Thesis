NAME1 = ["Liam","Olivia","Nathan","Ethan","Ethan","Jamie"] # "Evelyn","Emma","Evelyn","Chloe","Owen","Zoe","Charlotte","Alexander","Sam"
NAME2 = ["Ella","Taylor","Emma","Evelyn","Chloe","Owen"] # ,"Oliver","Nathan","Ethan","Ethan","Jamie","Lucas","Sam","Noah","Nathan"
LOGPROBS1 = [[" B",-3.8947296142578125],[" Nothing",-3.168689727783203],[" picture",-3.676868200302124],[" pencil",-4.247325897216797],[" What",-2.3582687377929688]]
LOGPROBS2 = [[" post(card)",-1.0898510217666626],[" What",-2.9725236892700195],[" Answer",-3.3187217712402344],[" post(card)",-0.36426877975463867],[" post(card)",-2.4164581298828125]]

# [[,],[,],[,],[,],[,]]

Liam_1 = [[" B",-3.8947296142578125],[" Nothing",-3.168689727783203],[" picture",-3.676868200302124],[" pencil",-4.247325897216797],[" What",-2.3582687377929688]]
Ella_1 = [[" post(card)",-1.0898510217666626],[" What",-2.9725236892700195],[" Answer",-3.3187217712402344],[" post(card)",-0.36426877975463867],[" post(card)",-2.4164581298828125]]


Olivia_2 = [[" pencil",-3.224931240081787],[" her (sketchpad)",-3.2337234020233154],[" None",-3.7472310066223145],[" sketch(pad)",-1.3050816059112549],[" None",-3.7472310066223145]]
Taylor_2 = [[" notebook",-2.1753485202789307],[" notebook",-2.3255653381347656],[" notebook",-2.0519113540649414],[" sketch(pad)",-1.1068676710128784],[" sketch",-1.1068676710128784]]
### sotry 2 interesting factor last answer in olivia 


Nathan_3 = [["None",-2.204036235809326],[" Answer",-3.937535285949707],[" bottle",-1.8936560153961182],[" bottle",-1.8274317979812622],[" water",-2.5616276264190674]]
Emma_3 = [[" Water",-0.9742696285247803],[" Water",-0.9979314208030701],[" Water",-0.9979314208030701],[" Water",-0.9979314208030701],[" What(nonsense answer)",-3.3570032119750977]]

Ethan_4 = [[" bag",-2.498054265975952],[" book",-1.809983253479004],[" book",-2.269308090209961],[" notebook",-3.07547664642334],[" to(-do bag)",-3.5168399810791016]]
Evelyn_4 = [[" T(ote bag)",-2.9456121921539307],[" book",-1.5605531930923462],[" tote(bag)",-2.589167833328247],[" B(nonsensce answer)",-3.2588279247283936],[" B(nonsence answer)",-3.2588279247283936]]

Ethan_5 = [[" A(answered alphabet)",-1.2437244653701782],[" silver(hair pin)",-3.7018256187438965],[" hair(pin)",-2.2796683311462402],[" hair(pin)",-2.2796683311462402],[" What(answered w question)",-2.3155274391174316]]
Chloe_5 = [[" hair(pin)",-0.689536452293396],[" A(nonsence answer)",-1.8592290878295898],[" hair(pin)",-0.4817523956298828],[" hair(pin)",-0.4123537838459015],[" Hair(pin)",-2.614521026611328]]
### chloe run 2 interresting fidning random sampler makes it output bad answer ... also first token has a lot lower probability

Jamie_6 = [[" bag(of coffe)",-3.606377363204956],[" Answer(fabricated answer)",-4.428120136260986],[" small(chip)",-5.3176469802856445],["(B + 1)(fabricated answer)",-3.486952543258667],[" What(question answer)",-2.5020785331726074]]
Owen_6 = [[" What(fabricated answer)",-2.685220718383789],["Mug.",-1.7586616277694702],[" Coffee",-2.640620708465576],[" What(fabricated answer)",-2.6892523765563965],[" Mug",-1.7586616277694702]]


LOGPROBS1 = Liam_1 + Olivia_2 + Nathan_3 + Ethan_4 + Ethan_5 + Jamie_6
LOGPROBS2 =  Ella_1 + Taylor_2 + Emma_3 + Evelyn_4 + Chloe_5 + Owen_6