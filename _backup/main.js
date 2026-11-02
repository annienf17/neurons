/**
 * EGOEXPLAINED — NEURAL JOURNEY
 * Vanilla JavaScript + Three.js / WebGL. No React, no build tooling.
 *
 * Scroll controls an actual 3D camera; neural networks are procedurally generated.
 * Cursor's colourful fluid wake is a second lightweight 2D canvas.
 * Edit CONFIG to change scene density, speed/length, bloom and colors.
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const CONFIG = {
  seed: 78941,
  journeyDistance: 268,   // how far the camera flies as the page scrolls
  fogDensity: 0.0075,
  bloomStrength: 1.18,
  bloomRadius: 0.68,
  bloomThreshold: 0.06,
  neuronsDesktop: 155,
  neuronsMobile: 87,
  dustDesktop: 2800,
  dustMobile: 1250,
  maxPixelRatioDesktop: 1.45,
  maxPixelRatioMobile: 1.0,
};

const $ = (selector) => document.querySelector(selector);
const webglCanvas = $('#webgl');
const inkCanvas = $('#ink');
const stage = $('#stage');
const journey = $('#journey');
const loading = $('#loading');
const loadingText = $('#loading-text');
const progressFill = $('#progress-fill');
const progressLabel = $('#progress-label');
const depthLabel = $('#coordinates-depth');
const chapters = [...document.querySelectorAll('[data-chapter]')];
const jumpButtons = [...document.querySelectorAll('[data-jump]')];
const motionButton = $('#motion-toggle');
const motionLabel = $('#motion-label');
const cursorOrb = $('#cursor-orb');

const mobile = window.matchMedia('(max-width: 760px)').matches;
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let reducedMotion = prefersReducedMotion.matches;
let targetProgress = 0;
let smoothProgress = 0;
let chapterIndex = -1;
let internalTime = 0;
let pointerX = 0;
let pointerY = 0;
let targetPointerX = 0;
let targetPointerY = 0;
let width = 1;
let height = 1;
let dpr = 1;

function seedRandom(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const random = seedRandom(CONFIG.seed);
const rand = (a, b) => a + (b - a) * random();
const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
const mix = (a, b, t) => a + (b - a) * t;
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const basePalette = [
  new THREE.Color('#fa297f'), new THREE.Color('#44eaff'),
  new THREE.Color('#9465ff'), new THREE.Color('#ffbf7d'),
  new THREE.Color('#5476ff'), new THREE.Color('#f054c8'),
];
const pickColor = () => basePalette[Math.floor(random() * basePalette.length)];

// =====================================
// THREE.JS / CINEMATIC WORLD
// =====================================
let renderer, scene, camera, composer;
let bloomPass;
let lineMaterial, particleMaterial;
let portalGroup;
let auroraMaterials = [];
let driftingGroup;
let clock = new THREE.Clock();
const lookSmooth = V(0, 0, -13);
const cameraSmooth = V(0, 0, 18);

function laneX(z) {
  return Math.sin(-z * 0.025) * 1.8 + Math.sin(-z * 0.063) * .7;
}
function laneY(z) {
  return Math.sin(-z * 0.038) * .84 + Math.cos(-z * 0.017) * .5;
}
function makeRenderer() {
  renderer = new THREE.WebGLRenderer({
    canvas: webglCanvas,
    antialias: false,
    alpha: false,
    depth: true,
    stencil: false,
    powerPreference: 'high-performance',
  });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.38;
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#03050d');
  scene.fog = new THREE.FogExp2('#03050d', CONFIG.fogDensity);
  camera = new THREE.PerspectiveCamera(mobile ? 69 : 65, 1, 0.06, 200);
  camera.position.set(0, 0, 18);

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    mobile ? .94 : CONFIG.bloomStrength,
    CONFIG.bloomRadius,
    CONFIG.bloomThreshold,
  );
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());
}

// Draw a curved neural fibre as many connected line segments. Packed into a
// SINGLE BufferGeometry for the entire scene, rather than thousands of draw calls.
function buildNeuralFibres() {
  const pos = [];
  const colours = [];
  const flows = [];
  const phases = [];
  const nodeLocations = [];
  const nodeColours = [];
  const addSegment = (p1, p2, c1, c2, a, b, phase) => {
    pos.push(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
    colours.push(c1.r, c1.g, c1.b, c2.r, c2.g, c2.b);
    flows.push(a, b);
    phases.push(phase, phase);
  };
  function addBend(from, to, colour, phase, bends = 11, opacityFalloff = 0) {
    const delta = to.clone().sub(from);
    const drift = V(rand(-2.4, 2.4), rand(-2.1, 2.1), rand(-1.9, 1.9));
    const ctrl = from.clone().addScaledVector(delta, .5).add(drift);
    const curve = new THREE.QuadraticBezierCurve3(from, ctrl, to);
    const dark = colour.clone().multiplyScalar(mix(.55, .25, opacityFalloff));
    let previous = curve.getPoint(0);
    for (let j = 1; j <= bends; j++) {
      const t = j / bends;
      const next = curve.getPoint(t);
      const c0 = colour.clone().lerp(dark, t * .78);
      const c1 = colour.clone().lerp(dark, t * .85);
      addSegment(previous, next, c0, c1, (j - 1) / bends, t, phase);
      previous = next;
    }
  }
  const total = mobile ? CONFIG.neuronsMobile : CONFIG.neuronsDesktop;
  for (let i = 0; i < total; i++) {
    const z = rand(-CONFIG.journeyDistance - 38, 12);
    const theta = rand(0, Math.PI * 2);
    const radius = rand(7.3, 23.5);
    const centre = V(
      laneX(z) + Math.cos(theta) * radius,
      laneY(z) + Math.sin(theta) * radius * .75,
      z,
    );
    const colour = pickColor().clone();
    const secondary = basePalette[Math.floor(random() * basePalette.length)].clone();
    nodeLocations.push(centre);
    nodeColours.push(colour);
    const arms = Math.floor(rand(4, 8));
    for (let k = 0; k < arms; k++) {
      const azimuth = (k / arms) * Math.PI * 2 + rand(-.44, .44);
      const phi = rand(-1.1, 1.1);
      const length = rand(4, 11.5);
      const outward = V(
        Math.cos(azimuth) * Math.cos(phi),
        Math.sin(azimuth) * Math.cos(phi),
        Math.sin(phi),
      );
      const tip = centre.clone().addScaledVector(outward, length);
      tip.x += rand(-1.4, 1.4);
      tip.y += rand(-1, 1);
      const phase = rand(0, 50);
      addBend(centre, tip, colour, phase, 10);
      const forks = random() < .77 ? 2 : 1;
      for (let n = 0; n < forks; n++) {
        const twig = tip.clone().addScaledVector(outward, rand(2.4, 6.5));
        twig.add(V(rand(-3.3, 3.3), rand(-3.4, 3.4), rand(-2.8, 2.8)));
        addBend(tip, twig, colour.clone().lerp(secondary, .39), phase + .7 * n, 7, .5);
        if (random() < .34) {
          const tiny = twig.clone().add(V(rand(-2.9, 2.9), rand(-2.7, 2.7), rand(-2.7, 2.7)));
          addBend(twig, tiny, colour, phase + n, 5, .95);
        }
      }
    }
  }

  // Synaptic bridges: nearby somas connect into a recognizable, woven network.
  for (let i = 0; i < nodeLocations.length; i++) {
    const a = nodeLocations[i];
    let links = 0;
    for (let j = i + 1; j < nodeLocations.length && links < 3; j++) {
      const b = nodeLocations[j];
      const d = a.distanceTo(b);
      if (d > 7 && d < 23 && random() < .085) {
        addBend(a, b, nodeColours[i].clone().lerp(nodeColours[j], .5), rand(0, 50), 17, .42);
        links++;
      }
    }
  }

  // A few long helically winding neural highways emphasize forward depth.
  for (let lane = 0; lane < 9; lane++) {
    const angle = (lane / 9) * Math.PI * 2;
    const r = rand(7.6, 12.8);
    const colour = basePalette[lane % basePalette.length].clone().multiplyScalar(.64);
    for (let zz = 20; zz > -CONFIG.journeyDistance - 35; zz -= 1.65) {
      const coordinates = (z) => {
        const th = angle + z * (lane % 2 ? .026 : -.021);
        return V(laneX(z) + Math.cos(th) * (r + 1.4 * Math.sin(z * .035 + lane)),
          laneY(z) + Math.sin(th) * (r * .73), z);
      };
      addSegment(coordinates(zz), coordinates(zz - 1.65), colour, colour, zz / 40, (zz - 1.65) / 40, lane * 4.4);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
  geometry.setAttribute('aFlow', new THREE.Float32BufferAttribute(flows, 1));
  geometry.setAttribute('aPhase', new THREE.Float32BufferAttribute(phases, 1));
  lineMaterial = new THREE.ShaderMaterial({
    uniforms: {uTime: {value: 0}, uProgress: {value: 0}},
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexColors: true,
    vertexShader: `
      attribute float aFlow;
      attribute float aPhase;
      varying vec3 vColor;
      varying float vFlow;
      varying float vPhase;
      varying float vDepth;
      uniform float uTime;
      void main() {
        vColor = color;
        vFlow = aFlow;
        vPhase = aPhase;
        vec3 p = position;
        p.xy += vec2(sin(uTime*.34 + p.z*.028 + aPhase), cos(uTime*.22 + p.z*.02 + aPhase)) * .055;
        vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
        vDepth = -mvPosition.z;
        gl_Position = projectionMatrix * mvPosition;
      }`,
    fragmentShader: `
      uniform float uTime;
      uniform float uProgress;
      varying vec3 vColor;
      varying float vFlow;
      varying float vPhase;
      varying float vDepth;
      void main() {
        float beat = pow(max(0.0, sin(vFlow * 13.0 - uTime * 2.65 + vPhase)), 16.0);
        float wave = .45 + .55 * sin(uTime * .65 + vPhase * 2.3);
        float visible = (1.0 - smoothstep(27.0, 137.0, vDepth)) * smoothstep(1.0, 7.0, vDepth);
        vec3 shift = mix(vec3(1.0), vec3(.7,1.25,1.35), smoothstep(.3,.95,uProgress)*.46);
        float intensity = (.46 + .26 * wave + beat * 2.35);
        gl_FragColor = vec4(vColor * shift * intensity, (0.28 + beat * .71) * visible);
      }`,
  });
  const network = new THREE.LineSegments(geometry, lineMaterial);
  network.frustumCulled = false;
  scene.add(network);
  return {nodeLocations, nodeColours};
}

function buildParticles(nodes) {
  const pos = [], colours = [], sizes = [], phases = [], kinds = [];
  function add(p, c, size, kind) {
    pos.push(p.x,p.y,p.z);
    colours.push(c.r,c.g,c.b);
    sizes.push(size);
    phases.push(rand(0,Math.PI*2));
    kinds.push(kind);
  }
  nodes.nodeLocations.forEach((p, i) => {
    add(p, nodes.nodeColours[i], rand(7.5, 14), 1);
    for(let j=0;j<3;j++) {
      add(p.clone().add(V(rand(-1.9,1.9),rand(-1.9,1.9),rand(-1.9,1.9))),
        nodes.nodeColours[i].clone().lerp(new THREE.Color('#ffffff'), .25), rand(2.5,5), .4);
    }
  });
  const count = mobile ? CONFIG.dustMobile : CONFIG.dustDesktop;
  for(let i=0;i<count;i++) {
    const z = rand(-CONFIG.journeyDistance-50, 22);
    const a = rand(0,Math.PI*2);
    const r = Math.sqrt(random()) * 41 + 1.5;
    const position = V(laneX(z) + Math.cos(a)*r,laneY(z)+Math.sin(a)*r*.8,z);
    const c = random() < .57 ? basePalette[1] : random()<.6?basePalette[2]:basePalette[0];
    add(position,c,rand(.9,3.4),0);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
  geometry.setAttribute('aSize', new THREE.Float32BufferAttribute(sizes, 1));
  geometry.setAttribute('aPhase', new THREE.Float32BufferAttribute(phases, 1));
  geometry.setAttribute('aKind', new THREE.Float32BufferAttribute(kinds, 1));
  particleMaterial = new THREE.ShaderMaterial({
    uniforms: {uTime:{value:0},uPixelRatio:{value:1},uProgress:{value:0}},
    transparent:true,depthWrite:false,depthTest:true,vertexColors:true,
    blending:THREE.AdditiveBlending,
    vertexShader:`
      attribute float aSize;
      attribute float aPhase;
      attribute float aKind;
      uniform float uTime;
      uniform float uPixelRatio;
      varying vec3 vColor;
      varying float vPhase;
      varying float vDepth;
      varying float vKind;
      void main(){
        vColor = color;
        vPhase = aPhase;
        vKind = aKind;
        vec3 p = position;
        p.xy += vec2(sin(uTime*.34 + aPhase + p.z*.08), cos(uTime*.27 + aPhase + p.z*.05)) * (.06 + aKind * .05);
        vec4 mv = modelViewMatrix*vec4(p,1.0);
        vDepth = -mv.z;
        float twinkle = .76 + .24 * sin(uTime*2.1 + aPhase*3.8);
        gl_PointSize = clamp(uPixelRatio * aSize * 30.0 / max(1.0,-mv.z) * twinkle, 1.0, 58.0*uPixelRatio);
        gl_Position=projectionMatrix*mv;
      }`,
    fragmentShader:`
      uniform float uTime;
      uniform float uProgress;
      varying vec3 vColor;
      varying float vPhase;
      varying float vDepth;
      varying float vKind;
      void main(){
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float r = length(p);
        if(r>1.0) discard;
        float halo = pow(max(0.0,1.0-r), 2.9);
        float core = pow(max(0.0,1.0-r), 16.0);
        float fog = 1.0 - smoothstep(28.0, 140.0, vDepth);
        float shine = .77 + .23*sin(uTime*1.4+vPhase*5.0);
        float a = (halo * .63 + core*.7) * fog * shine;
        vec3 tint = mix(vColor,vec3(.78,1.0,1.0), core * .3);
        gl_FragColor = vec4(tint*(.76+core*1.5),a);
      }`,
  });
  const points = new THREE.Points(geometry,particleMaterial);
  points.frustumCulled = false;
  scene.add(points);

  // A few actual 3D structures; highlights reveal their faceted contours as you fly past.
  const ico = new THREE.IcosahedronGeometry(.37, 1);
  const instanced = new THREE.InstancedMesh(
    ico,
    new THREE.MeshBasicMaterial({color:0xffffff,vertexColors:false,transparent:true,opacity:.65,blending:THREE.AdditiveBlending,depthWrite:false}),
    Math.min(nodes.nodeLocations.length, mobile?65:115),
  );
  const dummy = new THREE.Object3D();
  for(let i=0;i<instanced.count;i++) {
    dummy.position.copy(nodes.nodeLocations[i]);
    dummy.scale.setScalar(rand(.36,1.8));
    dummy.rotation.set(rand(0,3),rand(0,3),rand(0,3));
    dummy.updateMatrix();
    instanced.setMatrixAt(i,dummy.matrix);
    instanced.setColorAt(i,nodes.nodeColours[i]);
  }
  instanced.instanceMatrix.needsUpdate=true;
  if(instanced.instanceColor) instanced.instanceColor.needsUpdate=true;
  scene.add(instanced);
}

// Procedural colour veils, no textures/videos/assets required.
function buildAuroras() {
  for(let i=0;i<7;i++) {
    const z = 2-i*49;
    const geometry=new THREE.PlaneGeometry(120,80,1,1);
    const material = new THREE.ShaderMaterial({
      uniforms:{uTime:{value:0},uProgress:{value:0},uSeed:{value:i*1.7}},
      vertexShader:`
        varying vec2 vUv;
        void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
      fragmentShader:`
        uniform float uTime;
        uniform float uProgress;
        uniform float uSeed;
        varying vec2 vUv;
        void main(){
          vec2 uv=vUv;
          float bend=sin(uv.x*7.0+uSeed+uTime*.13)*.10
                    +sin(uv.x*13.0-uSeed*.7-uTime*.08)*.034;
          float y=uv.y-.5-bend;
          float b1=exp(-pow(y-.11,2.0)*31.0);
          float b2=exp(-pow(y+.18+sin(uv.x*4.0+uTime*.12)*.08,2.0)*35.0);
          float wisps=.65+.35*sin(uv.x*20.0 + sin(uv.y*11.0)*3.0 + uSeed + uTime*.19);
          float edge=pow(sin(uv.x*3.14159),1.7)*pow(sin(uv.y*3.14159),1.5);
          vec3 pink=vec3(.92,.025,.37), blue=vec3(.025,.47,.90), violet=vec3(.40,.08,.92);
          vec3 c=mix(violet,blue,uv.x+sin(uSeed)*.25);
          c=mix(c,pink, .34 + .34*sin(uv.x*6.0+uSeed+uProgress*4.0));
          float a=(b1*.14+b2*.12)*edge*wisps;
          gl_FragColor=vec4(c*2.2,a);
        }`,
      transparent:true,blending:THREE.AdditiveBlending,depthWrite:false,depthTest:true,
      side:THREE.DoubleSide,
    });
    const plane=new THREE.Mesh(geometry,material);
    plane.position.set(rand(-9,9),rand(-2,2),z);
    plane.rotation.z=rand(-.25,.25);
    scene.add(plane);
    auroraMaterials.push(material);
  }
}

function buildPortals() {
  portalGroup=new THREE.Group();
  for(let i=0;i<9;i++) {
    const z= -26-i*30.5;
    const radius=rand(4.8,8.5);
    const colour=basePalette[i%basePalette.length].clone();
    for(let ring=0;ring<2;ring++){
      const n=160;
      const vertices=[];
      const ringRadius=radius+(ring*.75);
      for(let j=0;j<=n;j++){
        const t=(j/n)*Math.PI*2;
        const wave=Math.sin(t*6+i*2)*.3 + Math.cos(t*9+i)*.1;
        vertices.push(
          laneX(z)+Math.cos(t)* (ringRadius+wave),
          laneY(z)+Math.sin(t)* (ringRadius+wave)*.8,
          z+Math.sin(3*t+i)*2.1+ring*.8,
        );
      }
      const lineGeometry=new THREE.BufferGeometry();
      lineGeometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
      const line=new THREE.Line(lineGeometry,new THREE.LineBasicMaterial({
        color:colour,transparent:true,opacity:ring===0?.31:.13,
        blending:THREE.AdditiveBlending,depthWrite:false,
      }));
      portalGroup.add(line);
    }
  }
  scene.add(portalGroup);
}

// =====================================
// WATER-LIKE CURSOR / IRIDESCENT WAKE
// =====================================
const ink = inkCanvas.getContext('2d', {alpha:true});
const trails=[];
const ripples=[];
let lastPointer=null;
let orbX=-100,orbY=-100;
let activePointer=false;
let lastTrailAt=0;

function handlePointer(event) {
  if (event.pointerType === 'touch' || reducedMotion) return;
  const rect=stage.getBoundingClientRect();
  const x=event.clientX-rect.left;
  const y=event.clientY-rect.top;
  targetPointerX=clamp((x/width-.5)*2,-1,1);
  targetPointerY=clamp((y/height-.5)*2,-1,1);
  activePointer=true;
  if(lastPointer) {
    const dx=x-lastPointer.x;
    const dy=y-lastPointer.y;
    const d=Math.hypot(dx,dy);
    if(d>1.8 && performance.now()-lastTrailAt>12) {
      // Interpolate the wake for fast mouse movement; this makes it feel like a brush.
      const steps=Math.min(8,Math.ceil(d/16));
      for(let s=1;s<=steps;s++) {
        const t=s/steps;
        trails.push({
          x:mix(lastPointer.x,x,t), y:mix(lastPointer.y,y,t),
          age:0, vx:dx*.019+rand(-.38,.38), vy:dy*.014+rand(-.38,.38),
          hue:(190+progressToHue(smoothProgress)+rand(-36,60))%360,
          radius:rand(1.2,3.3),
        });
      }
      lastTrailAt=performance.now();
    }
  }
  lastPointer={x,y};
}
function progressToHue(progress){return Math.sin(progress*Math.PI*2)*36;}
stage.addEventListener('pointermove',handlePointer,{passive:true});
stage.addEventListener('pointerleave',()=>{
  activePointer=false;
  lastPointer=null;
  targetPointerX=targetPointerY=0;
  cursorOrb.style.opacity='0';
},{passive:true});
stage.addEventListener('pointerenter',()=>{if(!reducedMotion)cursorOrb.style.opacity='1';});
stage.addEventListener('pointerdown',(e)=>{
  if(reducedMotion || e.pointerType==='touch')return;
  const rect=stage.getBoundingClientRect();
  ripples.push({x:e.clientX-rect.left,y:e.clientY-rect.top,age:0});
});

function animateCursor(dt) {
  ink.clearRect(0,0,width,height);
  if(reducedMotion) return;
  if(activePointer && lastPointer) {
    const follow=1-Math.exp(-dt*14);
    orbX=mix(orbX<0?lastPointer.x:orbX,lastPointer.x,follow);
    orbY=mix(orbY<0?lastPointer.y:orbY,lastPointer.y,follow);
    cursorOrb.style.transform=`translate3d(${orbX-10.5}px,${orbY-10.5}px,0)`;
  }
  // Soft fluid glow follows the spline of remembered cursor positions.
  for(let i=trails.length-1;i>=0;i--) {
    const p=trails[i];
    p.age+=dt;
    p.vx*=Math.exp(-dt*2.1);
    p.vy*=Math.exp(-dt*2.1);
    p.x+=p.vx*dt*37 + Math.sin(p.age*7+p.hue)*dt*1.5;
    p.y+=p.vy*dt*37 + Math.cos(p.age*6+p.hue)*dt*1.5;
    if(p.age>1.15) trails.splice(i,1);
  }
  // Keep the brightest cores, tinted by hue, in an additive glow field.
  ink.save();
  ink.globalCompositeOperation='lighter';
  for(let i=0;i<trails.length;i++) {
    const p=trails[i];
    const life=Math.pow(1-p.age/1.15,1.9);
    if(life<=0)continue;
    const r=p.radius*(2+life*4);
    const grad=ink.createRadialGradient(p.x,p.y,0,p.x,p.y,r*2.6);
    grad.addColorStop(0,`hsla(${p.hue},100%,76%,${life*.44})`);
    grad.addColorStop(.24,`hsla(${(p.hue+66)%360},100%,61%,${life*.2})`);
    grad.addColorStop(1,`hsla(${p.hue},100%,50%,0)`);
    ink.fillStyle=grad;
    ink.beginPath();
    ink.arc(p.x,p.y,r*2.6,0,Math.PI*2);
    ink.fill();
  }
  // Glassy connection between fresh points: the 'brush on water' surface.
  const recent=trails.filter(p=>p.age<.4);
  for(let pass=0;pass<2;pass++) {
    ink.lineWidth=pass===0?10:2.3;
    ink.lineCap='round';
    ink.lineJoin='round';
    ink.strokeStyle=pass===0?'rgba(171,86,255,.065)':'rgba(136,247,255,.21)';
    if(recent.length>2) {
      ink.beginPath();
      ink.moveTo(recent[0].x,recent[0].y);
      for(let i=1;i<recent.length-1;i++) {
        const p=recent[i],q=recent[i+1];
        ink.quadraticCurveTo(p.x,p.y,(p.x+q.x)*.5,(p.y+q.y)*.5);
      }
      ink.stroke();
    }
  }
  for(let i=ripples.length-1;i>=0;i--) {
    const r=ripples[i];r.age+=dt;
    const alpha=Math.pow(Math.max(0,1-r.age/.9),2);
    if(r.age>.9){ripples.splice(i,1);continue;}
    ink.lineWidth=1.1;
    ink.strokeStyle=`rgba(115,233,255,${alpha*.5})`;
    ink.beginPath();ink.arc(r.x,r.y,12+r.age*115,0,Math.PI*2);ink.stroke();
    ink.strokeStyle=`rgba(255,61,162,${alpha*.3})`;
    ink.beginPath();ink.arc(r.x,r.y,4+r.age*70,0,Math.PI*2);ink.stroke();
  }
  ink.restore();
}

// =====================================
// SCROLL-DRIVEN 3D CAMERA + UI
// =====================================
function readScroll() {
  const distance=Math.max(1,journey.offsetHeight-window.innerHeight);
  const offset=window.scrollY + journey.getBoundingClientRect().top; // absolute top
  targetProgress=clamp((window.scrollY-offset)/distance,0,1);
}
function switchChapter(index) {
  if(index===chapterIndex)return;
  chapterIndex=index;
  chapters.forEach((c,i)=>{
    c.classList.toggle('is-active',i===index);
    c.setAttribute('aria-hidden',String(i!==index));
    c.querySelectorAll('a,button').forEach(el => { el.tabIndex = i === index ? 0 : -1; });
  });
  jumpButtons.forEach((b,i)=>{
    b.classList.toggle('is-current',i===index);
    if(i===index)b.setAttribute('aria-current','step');
    else b.removeAttribute('aria-current');
  });
}
function updateHud(progress) {
  const number=Math.round(progress*100);
  progressFill.style.width=`${number}%`;
  progressLabel.textContent=`${String(number).padStart(2,'0')}%`;
  depthLabel.textContent=`DEPTH ${(progress*CONFIG.journeyDistance).toFixed(1).padStart(5,'0')}`;
  switchChapter(Math.min(4,Math.floor(progress*5)));
}
function moveToChapter(index) {
  const distance=Math.max(1,journey.offsetHeight-window.innerHeight);
  const destinations=[0,.22,.43,.65,.86];
  const absoluteY=window.scrollY+journey.getBoundingClientRect().top;
  window.scrollTo({top:absoluteY+distance*destinations[index],behavior:reducedMotion?'auto':'smooth'});
}
jumpButtons.forEach(btn=>btn.addEventListener('click',()=>moveToChapter(Number(btn.dataset.jump))));
function setMotion(enabled){
  reducedMotion=!enabled;
  motionButton.setAttribute('aria-pressed',String(reducedMotion));
  motionLabel.textContent=enabled?'MOTION ON':'MOTION OFF';
  if(reducedMotion){trails.length=0;ripples.length=0;cursorOrb.style.opacity='0';}
}
motionButton.addEventListener('click',()=>setMotion(reducedMotion));
setMotion(!reducedMotion);
prefersReducedMotion.addEventListener?.('change',(event)=>setMotion(!event.matches));
window.addEventListener('scroll',readScroll,{passive:true});

function resize() {
  width=Math.max(1,window.innerWidth);
  height=Math.max(1,window.innerHeight);
  dpr=Math.min(window.devicePixelRatio||1,mobile?CONFIG.maxPixelRatioMobile:CONFIG.maxPixelRatioDesktop);
  renderer.setPixelRatio(dpr);
  renderer.setSize(width,height,false);
  composer.setPixelRatio(dpr);
  composer.setSize(width,height);
  camera.aspect=width/height;
  camera.fov=width<760?69:65;
  camera.updateProjectionMatrix();
  inkCanvas.width=Math.round(width);
  inkCanvas.height=Math.round(height);
  inkCanvas.style.width=width+'px';
  inkCanvas.style.height=height+'px';
  particleMaterial.uniforms.uPixelRatio.value=dpr;
  readScroll();
}
function animate(){
  requestAnimationFrame(animate);
  const dt=Math.min(clock.getDelta(),.06);
  // Camera inertia keeps scroll-driven movement smooth even with wheel steps.
  const cameraDamping=reducedMotion?1:1-Math.exp(-dt*2.6);
  const pointerDamping=1-Math.exp(-dt*3.6);
  smoothProgress=mix(smoothProgress,targetProgress,cameraDamping);
  pointerX=mix(pointerX,targetPointerX,pointerDamping);
  pointerY=mix(pointerY,targetPointerY,pointerDamping);
  const z=18-smoothProgress*CONFIG.journeyDistance;
  const x=laneX(z)+pointerX*(mobile?.1:.76);
  const y=laneY(z)-pointerY*(mobile?.05:.48);
  const desired=V(x,y,z);
  const toward=V(laneX(z-13)+pointerX*.16,laneY(z-13)-pointerY*.11,z-16);
  cameraSmooth.lerp(desired,cameraDamping);
  lookSmooth.lerp(toward,cameraDamping);
  camera.position.copy(cameraSmooth);
  camera.lookAt(lookSmooth);
  camera.rotateZ(pointerX*.013);
  if(!reducedMotion)internalTime+=dt;
  lineMaterial.uniforms.uTime.value=internalTime;
  lineMaterial.uniforms.uProgress.value=smoothProgress;
  particleMaterial.uniforms.uTime.value=internalTime;
  particleMaterial.uniforms.uProgress.value=smoothProgress;
  for(const material of auroraMaterials){
    material.uniforms.uTime.value=internalTime;
    material.uniforms.uProgress.value=smoothProgress;
  }
  if(!document.hidden){
    composer.render(dt);
    animateCursor(dt);
  }
  updateHud(smoothProgress);
}

// =====================================
// START
// =====================================
try {
  if(!ink) throw new Error('Your browser does not support Canvas 2D.');
  makeRenderer();
  buildAuroras();
  const nodes=buildNeuralFibres();
  buildParticles(nodes);
  buildPortals();
  resize();
  window.addEventListener('resize',resize,{passive:true});
  window.addEventListener('orientationchange',resize,{passive:true});
  readScroll();
  smoothProgress=targetProgress;
  loadingText.textContent='Połączenia gotowe — rozpocznij podróż.';
  animate();
  // Remove loader after two frames, when the WebGL output is ready.
  requestAnimationFrame(()=>requestAnimationFrame(()=>loading.classList.add('is-ready')));
} catch(error) {
  console.error('Neural Journey:',error);
  loadingText.textContent='Nie udało się uruchomić WebGL. Sprawdź obsługę grafiki 3D i konsolę przeglądarki.';
  $('.loading__bar').style.display='none';
}
