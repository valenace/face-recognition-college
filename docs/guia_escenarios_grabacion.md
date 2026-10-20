# Guia de Escenarios de Grabacion para Pruebas del Sistema de Asistencia Facial

## Objetivo
Recopilar videos que cubran TODOS los escenarios posibles para validar:
1. **Registro** (enrolamiento) robusto
2. **Reconocimiento** en condiciones reales del salon
3. **Anti-fraude** (liveness) contra ataques reales
4. **Robustez social** (comportamientos humanos en un salon)
5. **Anti-alucinacion** (que no reconozca a quien no debe)

---

## PARTE 1: REGISTRO (Enrolamiento)

> Cada persona debe grabar estos videos para su propio registro.
> Camara: la misma que se usara en produccion (o webcam 720p+).
> Formato: MP4, minimo 720p, 30fps.

### 1.1 Iluminacion

| # | Escenario | Descripcion | Duracion |
|---|-----------|-------------|----------|
| R-L1 | Luz natural buena | Frente a ventana, luz uniforme en el rostro | 15 seg |
| R-L2 | Luz artificial oficina | Tubos fluorescentes tipicos de salon | 15 seg |
| R-L3 | Luz baja | Salon con luz tenue, atardecer | 15 seg |
| R-L4 | Contraluz | Ventana detras de la persona (silueta) | 10 seg |
| R-L5 | Luz lateral fuerte | Luz solo de un lado (sombra mitad rostro) | 10 seg |
| R-L6 | Mezcla luz natural+artificial | Combinacion tipica de salon real | 15 seg |

### 1.2 Angulos del rostro

| # | Escenario | Descripcion | Duracion |
|---|-----------|-------------|----------|
| R-A1 | Frontal directo | Mirando a camara de frente | 10 seg |
| R-A2 | Giro izquierda 30° | Rostro girado ligeramente a la izquierda | 10 seg |
| R-A3 | Giro derecha 30° | Rostro girado ligeramente a la derecha | 10 seg |
| R-A4 | Inclinacion arriba | Mirando ligeramente hacia arriba (como viendo pizarra) | 10 seg |
| R-A5 | Inclinacion abajo | Mirando ligeramente hacia abajo (como escribiendo) | 10 seg |
| R-A6 | Perfil parcial 45° | Medio perfil izquierdo | 10 seg |
| R-A7 | Perfil parcial 45° | Medio perfil derecho | 10 seg |

### 1.3 Accesorios y apariencia

| # | Escenario | Descripcion | Duracion |
|---|-----------|-------------|----------|
| R-C1 | Sin accesorios | Rostro limpio, sin nada | 10 seg |
| R-C2 | Lentes opticos | Con lentes de receta transparentes | 10 seg |
| R-C3 | Lentes de sol | Lentes oscuros (si la persona los usa regularmente) | 10 seg |
| R-C4 | Cabello suelto largo | Si aplica, cabello cubriendo parte del rostro | 10 seg |
| R-C5 | Cabello recogido | Si aplica, cabello amarrado | 10 seg |
| R-C6 | Gorra/sombrero | Con visera que genera sombra en ojos | 10 seg |
| R-C7 | Barba/bigote | Si aplica, con vello facial | 10 seg |
| R-C8 | Sin barba | Si aplica, afeitado (para personas que alternan) | 10 seg |

### 1.4 Expresiones faciales

| # | Escenario | Descripcion | Duracion |
|---|-----------|-------------|----------|
| R-E1 | Neutra | Sin expresion, relajado | 10 seg |
| R-E2 | Sonriendo | Sonrisa natural (cambia geometria del rostro) | 10 seg |
| R-E3 | Hablando | Movimiento de boca como si hablara | 10 seg |
| R-E4 | Ojos semi-cerrados | Como si tuviera sueno | 10 seg |
| R-E5 | Con bostezo | Boca abierta grande | 5 seg |
| R-E6 | Ceno fruncido | Como concentrado/enojado | 10 seg |

### 1.5 Distancia y posicion

| # | Escenario | Descripcion | Duracion |
|---|-----------|-------------|----------|
| R-D1 | Cerca (30-50cm) | Rostro grande en frame | 10 seg |
| R-D2 | Media (1-2m) | Distancia tipica de escritorio | 15 seg |
| R-D3 | Lejos (3-5m) | Como si fuera camara de seguridad en esquina | 15 seg |
| R-D4 | Movimiento acercandose | De lejos a cerca progresivamente | 15 seg |
| R-D5 | Movimiento alejandose | De cerca a lejos progresivamente | 15 seg |

### 1.6 Movimiento natural

| # | Escenario | Descripcion | Duracion |
|---|-----------|-------------|----------|
| R-M1 | Quieto mirando a camara | Sin movimiento | 10 seg |
| R-M2 | Movimiento natural cabeza | Girando suavemente como si mirara alrededor | 15 seg |
| R-M3 | Asintiendo | Movimiento vertical de cabeza | 10 seg |
| R-M4 | Negando | Movimiento horizontal de cabeza | 10 seg |
| R-M5 | Parpadeo natural | Parpadeos normales durante 15 seg | 15 seg |

---

## PARTE 2: RECONOCIMIENTO EN VIVO (Asistencia)

> Estos videos simulan lo que la camara del salon veria durante una clase.
> Camara: posicion fija en esquina del salon (angulo de seguridad).
> Formato: MP4, 720p+, 30fps, minimo 30 segundos cada uno.

### 2.1 Individuales

| # | Escenario | Descripcion | Duracion |
|---|-----------|-------------|----------|
| V-I1 | Entrada normal | Persona entra al salon, se sienta, mira al frente | 30 seg |
| V-I2 | Sentado de frente | Persona sentada mirando hacia la camara/pizarra | 30 seg |
| V-I3 | Sentado de lado | Persona sentada de perfil a la camara | 30 seg |
| V-I4 | Escribiendo | Persona mirando hacia abajo, escribiendo en cuaderno | 30 seg |
| V-I5 | Usando laptop | Cara parcialmente tapada por pantalla de laptop | 30 seg |
| V-I6 | Apoyando la cabeza | Mano en la mejilla/meenton, como aburrido | 30 seg |
| V-I7 | Mirando al celular | Cabeza inclinada hacia abajo | 30 seg |
| V-I8 | Durmiendo | Cabeza baja o apoyada en escritorio | 30 seg |
| V-I9 | Levantandose | Persona se para y camina | 15 seg |
| V-I10 | Saliendo del salon | Persona camina hacia la puerta y sale | 15 seg |

### 2.2 Multiples personas

| # | Escenario | Descripcion | Duracion |
|---|-----------|-------------|----------|
| V-M1 | 2 personas juntas | Dos personas sentadas una al lado de la otra | 30 seg |
| V-M2 | 3 personas juntas | Tres personas en fila | 30 seg |
| V-M3 | 5+ personas | Grupo grande (simular salon parcial) | 60 seg |
| V-M4 | Una delante de otra | Persona tapa parcialmente a otra detras | 30 seg |
| V-M5 | Personas cruzando | Alguien pasa frente a alguien sentado | 30 seg |
| V-M6 | Entrada grupal | 3-5 personas entran juntas al salon | 30 seg |
| V-M7 | Salon lleno | Maxima capacidad, todos sentados | 60 seg |
| V-M8 | Personas moviendose | Varios moviendose, hablando, cambiando de lugar | 60 seg |

### 2.3 Condiciones del salon

| # | Escenario | Descripcion | Duracion |
|---|-----------|-------------|----------|
| V-S1 | Salon iluminado | Todas las luces encendidas | 30 seg |
| V-S2 | Salon oscuro | Solo luz del proyector/pantalla | 30 seg |
| V-S3 | Proyector encendido | Luz del proyector dando en rostros | 30 seg |
| V-S4 | Cambio de luz | Encender/apagar luces durante la clase | 30 seg |
| V-S5 | Ventana abierta | Luz natural cambiando (nubes) | 30 seg |
| V-S6 | Persiana semi-cerrada | Lineas de sombra en rostros | 30 seg |

### 2.4 Oclusiones (rostro parcialmente tapado)

| # | Escenario | Descripcion | Duracion |
|---|-----------|-------------|----------|
| V-O1 | Mano en la cara | Persona se toca la cara, se cubre la boca | 15 seg |
| V-O2 | Cabello cubriendo | Cabello largo cayendo sobre ojos/mejillas | 15 seg |
| V-O3 | Bufanda/panuelo | Cubriendo menton y boca | 15 seg |
| V-O4 | Mascarilla | Con tapabocas (si aplica) | 15 seg |
| V-O5 | Taza/botella | Bebiendo, objeto tapa parte inferior del rostro | 15 seg |
| V-O6 | Cuaderno al frente | Sosteniendo cuaderno cerca de la cara | 15 seg |
| V-O7 | Capucha | Sudadera con capucha puesta | 15 seg |
| V-O8 | Otro compañero | Persona asomandose detras de otra | 15 seg |

---

## PARTE 3: ATAQUES DE FRAUDE (Anti-Spoofing / Liveness)

> Estos videos VALIDAN que el sistema detecte intentos de fraude.
> Camara: misma posicion que produccion.
> El sistema debe marcar "FRAUDE DETECTADO" en todos estos casos.

### 3.1 Ataques con pantalla

| # | Ataque | Descripcion | Duracion |
|---|--------|-------------|----------|
| F-P1 | Foto en celular | Foto del rostro mostrada en pantalla de smartphone | 15 seg |
| F-P2 | Foto en tablet | Foto en pantalla de iPad/tablet | 15 seg |
| F-P3 | Foto en laptop | Foto mostrada en pantalla de laptop | 15 seg |
| F-P4 | Video en celular | Video del rostro reproducido en smartphone | 15 seg |
| F-P5 | Video en tablet | Video reproducido en tablet | 15 seg |
| F-P6 | Foto en monitor | Foto a pantalla completa en monitor externo | 15 seg |
| F-P7 | Deepfake en pantalla | Video generado por IA en pantalla (si disponible) | 15 seg |

### 3.2 Ataques con material impreso

| # | Ataque | Descripcion | Duracion |
|---|--------|-------------|----------|
| F-I1 | Foto impresa papel | Foto impresa en papel comun | 15 seg |
| F-I2 | Foto impresa glossy | Foto impresa en papel fotografico | 15 seg |
| F-I3 | Foto tamano real | Impresion a escala real del rostro | 15 seg |
| F-I4 | Foto agrandada | Impresion grande (carta/A4) sostenida al frente | 15 seg |
| F-I5 | Recorte de rostro | Solo la cara recortada de una foto | 15 seg |

### 3.3 Ataques con mascara/3D

| # | Ataque | Descripcion | Duracion |
|---|--------|-------------|----------|
| F-M1 | Mascara plana | Mascara impresa o de carton frente a la cara | 15 seg |
| F-M2 | Mascara 3D | Si disponible, mascara con relieve facial | 15 seg |
| F-M3 | Muneco/maniqui | Rostro de muneco frente a camara | 15 seg |

### 3.4 Ataques de suplantacion

| # | Ataque | Descripcion | Duracion |
|---|--------|-------------|----------|
| F-S1 | Gemelo/similar | Persona con rostro muy similar al registrado | 15 seg |
| F-S2 | Parecido familiar | Hermano/primo con rasgos similares | 15 seg |
| F-S3 | Maquillaje imitacion | Alguien maquillado para parecerse (si es factible) | 15 seg |
| F-S4 | Cirugia/plastico | Foto plastica moldeada al rostro (extremo) | 15 seg |

---

## PARTE 4: ESCENARIOS SOCIALES DEL SALON

> Comportamientos humanos tipicos que el sistema debe manejar sin fallar.

### 4.1 Comportamientos individuales

| # | Escenario | Que debe hacer el sistema | Duracion |
|---|-----------|--------------------------|----------|
| S-I1 | Estudiante llega tarde (5 min) | Registrarlo cuando sea visible | 30 seg |
| S-I2 | Estudiante llega muy tarde (30 min) | Registrarlo igual | 30 seg |
| S-I3 | Estudiante se va temprano | Mantener registro con hora de salida | 30 seg |
| S-I4 | Estudiante sale y vuelve | No duplicar registro, actualizar ultima vez visto | 45 seg |
| S-I5 | Estudiante cambia de asiento | Mantener identidad, actualizar posicion | 30 seg |
| S-I6 | Estudiante en ultima fila | Detectar rostro pequeno pero valido | 30 seg |
| S-I7 | Estudiante en primera fila | Detectar rostro grande, posible angulo picado | 30 seg |
| S-I8 | Estudiante bostezando | No confundir boca abierta con fraude | 15 seg |
| S-I9 | Estudiante estirandose | Movimiento brusco, no perder tracking | 15 seg |
| S-I10 | Estudiante riendose | Expresion extrema, mantener reconocimiento | 15 seg |

### 4.2 Comportamientos grupales

| # | Escenario | Que debe hacer el sistema | Duracion |
|---|-----------|--------------------------|----------|
| S-G1 | Entrada en masa | Registrar a todos los que entran juntos | 60 seg |
| S-G2 | Salida en masa (fin de clase) | No duplicar, cerrar registros limpios | 30 seg |
| S-G3 | Grupo trabajando junto | Varios rostros cerca, mantener IDs separados | 60 seg |
| S-G4 | Exposicion al frente | Persona de pie frente a todos, angulo diferente | 30 seg |
| S-G5 | Profesor caminando | No registrar al profesor como estudiante | 30 seg |
| S-G6 | Visita/observador | Persona no registrada, marcar como desconocido | 30 seg |
| S-G7 | Alumno filmando con celular | No confundir celular con ataque de fraude | 15 seg |
| S-G8 | Selfie grupal | No registrar como asistencia si es foto | 15 seg |

### 4.3 Situaciones extremas

| # | Escenario | Que debe hacer el sistema | Duracion |
|---|-----------|--------------------------|----------|
| S-X1 | Salon vacio | No registrar a nadie, no fallar | 30 seg |
| S-X2 | Solo profesor presente | Solo registrar si esta en la DB | 30 seg |
| S-X3 | Apagon (sin luz) | No detectar rostros, no alucinar | 30 seg |
| S-X4 | Camara movida/vibrando | No perder registros previos, re-adaptar | 30 seg |
| S-X5 | Persona entra a oscuras | Detectar cuando haya suficiente luz | 30 seg |
| S-X6 | Camara parcialmente obstruida | Detectar rostros visibles, ignorar zona tapada | 30 seg |

---

## PARTE 5: ANTI-ALUCINACION (Falsos Positivos)

> El sistema NO debe reconocer a quien no esta en la base de datos.

### 5.1 Rostros no registrados

| # | Escenario | Que debe hacer | Duracion |
|---|-----------|----------------|----------|
| A-N1 | Persona totalmente nueva | Marcar "Desconocido", no registrar | 30 seg |
| A-N2 | Persona de otra clase | Marcar "Desconocido" | 30 seg |
| A-N3 | Personal de limpieza | Marcar "Desconocido" | 15 seg |
| A-N4 | Padre de familia visita | Marcar "Desconocido" | 15 seg |

### 5.2 Objetos que parecen rostros (pareidolia)

| # | Escenario | Que debe hacer | Duracion |
|---|-----------|----------------|----------|
| A-P1 | Poster con cara en pared | NO detectar como rostro | 30 seg |
| A-P2 | Foto enmarcada en escritorio | NO detectar como rostro | 30 seg |
| A-P3 | Rostro en pantalla de laptop | NO detectar como rostro real | 30 seg |
| A-P4 | Reflejo en ventana/espejo | NO detectar como rostro real | 30 seg |
| A-P5 | Sombra con forma de cara | NO detectar como rostro | 30 seg |
| A-P6 | Estampado de camiseta con cara | NO detectar como rostro | 15 seg |
| A-P7 | Muneco/peluche con cara | NO detectar como rostro real | 15 seg |
| A-P8 | Dibujo en cuaderno con cara | NO detectar como rostro | 15 seg |

### 5.3 Condiciones que confunden al detector

| # | Escenario | Que debe hacer | Duracion |
|---|-----------|----------------|----------|
| A-C1 | Luz estroboscopica | No crear identidades fantasma | 15 seg |
| A-C2 | Reflejo de proyector en cara | No perder reconocimiento real | 15 seg |
| A-C3 | Humo/vapor (si aplica) | No detectar rostros en humo | 15 seg |
| A-C4 | Moscas/insectos frente a camara | No detectar como rostros | 10 seg |
| A-C5 | Hojas/papeles volando | No detectar como rostros | 10 seg |

### 5.4 Confusion entre personas registradas

| # | Escenario | Que debe hacer | Duracion |
|---|-----------|----------------|----------|
| A-R1 | Dos personas registradas juntas | Reconocer a ambas correctamente, no intercambiar | 30 seg |
| A-R2 | Persona registrada + no registrada | Reconocer solo a la registrada | 30 seg |
| A-R3 | 3+ personas registradas juntas | Reconocer a todas sin duplicar | 30 seg |
| A-R4 | Persona registrada sale, entra otra | Cambiar identidad del tracker correctamente | 30 seg |

---

## PARTE 6: ESTRES DEL SISTEMA (Performance)

### 6.1 Carga computacional

| # | Escenario | Metrica a medir | Duracion |
|---|-----------|-----------------|----------|
| P-C1 | 1 persona en frame | FPS, latencia de reconocimiento | 60 seg |
| P-C2 | 5 personas en frame | FPS, latencia por persona | 60 seg |
| P-C3 | 10 personas en frame | FPS, si se pierden detecciones | 60 seg |
| P-C4 | 20 personas en frame | FPS, estabilidad del tracker | 60 seg |
| P-C5 | Movimiento continuo masivo | Si el tracker mantiene IDs | 60 seg |

### 6.2 Duracion extendida

| # | Escenario | Metrica a medir | Duracion |
|---|-----------|-----------------|----------|
| P-D1 | Clase completa sin incidentes | Estabilidad de memoria, FPS promedio | 60 min |
| P-D2 | Clase con entradas/salidas | Consistencia de registros | 60 min |
| P-D3 | Dia completo de clases | Acumulacion de CSV, rendimiento | 8 horas |

---

## INSTRUCCIONES DE GRABACION

### Para REGISTRO (Parte 1):
1. Usar la misma camara que se usara en produccion
2. Grabar en MP4, minimo 720p
3. Cada video debe tener UN SOLO rostro visible
4. Nombrar archivos: `{ID}_R-{codigo}.mp4` (ej: `Lino313_R-L1.mp4`)
5. Combinar todos los clips de una persona en un solo video de registro

### Para RECONOCIMIENTO (Partes 2-6):
1. Camara en posicion fija (angulo de seguridad del salon)
2. NO editar el video (sin cortes, sin efectos)
3. Nombrar archivos: `TEST_{categoria}-{codigo}.mp4` (ej: `TEST_V-M3.mp4`)
4. Incluir un archivo `manifest.csv` con:
   ```
   archivo,personas_presentes,personas_que_deberia_reconocer,duracion
   TEST_V-M3.mp4,"Lino,Valu,Jorge","Lino,Valu,Jorge",60
   TEST_F-P1.mp4,"Lino mostrando foto","ninguno (fraude)",15
   ```

### Para FRAUDE (Parte 3):
1. Mostrar la foto/pantalla claramente frente a la camara
2. Mover la foto como si fuera una persona real
3. Intentar diferentes angulos y distancias
4. Probar con brillo de pantalla alto y bajo

### Para ANTI-ALUCINACION (Parte 5):
1. Colocar los objetos (posters, fotos, munecos) en posiciones realistas
2. Grabar minimo 30 segundos para ver si el detector se activa
3. Verificar que NO se generan filas en el CSV de asistencia

---

## CHECKLIST RAPIDO DE PRIORIDADES

### MINIMO VIABLE (grabar primero):
- [ ] R-L1, R-A1, R-E1 (registro basico por persona)
- [ ] V-I1, V-I2 (reconocimiento individual)
- [ ] V-M1, V-M2 (multiples personas)
- [ ] F-P1, F-P2 (fraude con pantalla)
- [ ] F-I1 (fraude con impresion)
- [ ] A-N1 (persona no registrada)
- [ ] A-P1, A-P3 (objetos que parecen rostros)

### PRIORIDAD ALTA (grabar segundo):
- [ ] R-L2, R-L3 (iluminacion variada)
- [ ] R-C1, R-C2 (accesorios)
- [ ] V-I4, V-I5, V-I6 (posiciones naturales)
- [ ] V-O1, V-O4 (oclusiones comunes)
- [ ] V-M3, V-M6 (grupo grande, entrada grupal)
- [ ] F-P4, F-P5 (video en pantalla)
- [ ] A-R1, A-R2 (confusion entre registrados)
- [ ] S-I1, S-I4 (llegada tarde, sale y vuelve)

### PRIORIDAD MEDIA (grabar tercero):
- [ ] Resto de escenarios de las Partes 1-5
- [ ] P-C1 a P-C3 (stress test basico)

### PRIORIDAD BAJA (si hay tiempo):
- [ ] Escenarios extremos (S-X*)
- [ ] Ataques avanzados de fraude
- [ ] Tests de duracion extendida (P-D*)
