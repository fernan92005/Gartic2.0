"""Controles sintéticos de mecanismo, ajenos a test/giro y a etiquetas reales."""
import unittest
from caracteristicas_b import caracteristicas_b,clauses,B1,B2,B3,B4


def case(sources,response,question=''):
    return dict(fuentes=sources,respuesta=response,pregunta=question,senales={})


class Matices(unittest.TestCase):
    def test_clausula_ajena(self):
        f=caracteristicas_b(case(['Alba abrirá 20 aulas; Berta solo abrirá 10 oficinas.'],'Alba abrirá 20 aulas.'))
        self.assertEqual(f['matiz_clausula_restrictivo'],0.)
    def test_matiz_cifra(self):
        f=caracteristicas_b(case(['Alba abrirá aproximadamente 20 aulas.'],'Alba abrirá 20 aulas.'))
        self.assertGreater(f['matiz_clausula_aproximado'],0.)
    def test_preservado_otra_oracion(self):
        f=caracteristicas_b(case(['Alba abrirá 20 aulas solo si se aprueba el plan.'],'Alba abrirá 20 aulas. Esto estará sujeto a aprobación.'))
        self.assertEqual(f['matiz_parafrasis_perdido'],0.)
    def test_hasta_temporal(self):
        f=caracteristicas_b(case(['El ERTE durará hasta el 31 de julio.'],'El ERTE durará hasta el 31 de julio.'))
        self.assertEqual(f['matiz_clausula_aproximado'],0.)
    def test_condicion_otra_entidad(self):
        f=caracteristicas_b(case(['Alba abrirá 20 aulas si se aprueba el plan.'],'Alba abrirá 20 aulas. Berta abrirá oficinas en caso de aprobación.'))
        self.assertGreater(f['matiz_parafrasis_perdido'],0.)
    def test_resumen_clausulas(self):
        f=caracteristicas_b(case(['Alba solo abrirá 20 aulas; Berta abrirá 10 oficinas.'],'Alba únicamente abrirá 20 aulas y Berta abrirá 10 oficinas.'))
        self.assertGreater(f['matiz_parafrasis_preservado'],0.)
    def test_parafrasis_fuerte(self):
        f=caracteristicas_b(case(['El centro solo imparte cursos técnicos.'],'El centro únicamente imparte cursos técnicos.'))
        self.assertEqual(f['matiz_parafrasis_perdido'],0.)
        self.assertGreater(f['matiz_parafrasis_preservado'],0.)
    def test_condicion_aprobacion(self):
        f=caracteristicas_b(case(['El centro abrirá 20 aulas si se aprueba el plan.'],'El centro abrirá 20 aulas en caso de aprobación.'))
        self.assertEqual(f['matiz_parafrasis_perdido'],0.)
    def test_propuesta_no_condicion(self):
        f=caracteristicas_b(case(['El centro abrirá 20 aulas si se aprueba el plan.'],'La propuesta del centro abrirá 20 aulas.'))
        self.assertGreater(f['matiz_parafrasis_perdido'],0.)
        self.assertGreater(f['matiz_equivalencia_incierta'],0.)
    def test_quedarian_no_condicion(self):
        f=caracteristicas_b(case(['El centro abrirá 20 aulas si se aprueba el plan.'],'En el centro quedarían 20 aulas.'))
        self.assertGreater(f['matiz_parafrasis_perdido'],0.)
    def test_prevision_no_certeza(self):
        f=caracteristicas_b(case(['El centro prevé abrir 20 aulas.'],'El centro abrirá 20 aulas.'))
        self.assertGreater(f['matiz_parafrasis_perdido'],0.)
    def test_parcial_no_mayoria(self):
        f=caracteristicas_b(case(['Algunos centros abrirán nuevas aulas.'],'La mayoría de centros abrirán nuevas aulas.'))
        self.assertGreater(f['matiz_parafrasis_perdido'],0.)
    def test_hasta_no_aproximado(self):
        f=caracteristicas_b(case(['El centro recibirá hasta 100 alumnos.'],'El centro recibirá aproximadamente 100 alumnos.'))
        self.assertGreater(f['matiz_parafrasis_perdido'],0.)
        self.assertGreater(f['matiz_equivalencia_incierta'],0.)
    def test_y_nominal(self):
        self.assertEqual(len(clauses('El centro de investigación y desarrollo abrirá tres laboratorios.')),1)
    def test_y_proposiciones(self):
        self.assertEqual(len(clauses('Alba abrirá aulas y Berta cerrará oficinas.')),2)
    def test_enumeracion_decimal(self):
        self.assertEqual(len(clauses('El centro tiene aulas, oficinas y 2,5 laboratorios.')),1)


class Cobertura(unittest.TestCase):
    def test_dos_fuentes(self):
        f=caracteristicas_b(case(['La biblioteca conserva volúmenes antiguos.','La biblioteca organiza talleres infantiles.'],'La biblioteca conserva volúmenes antiguos y organiza talleres infantiles.'))
        self.assertGreater(f['ganancia_cobertura_union'],0.)
        self.assertEqual(f['numero_fuentes_complementarias'],2.)
    def test_tres_fuentes(self):
        f=caracteristicas_b(case(['Alba conserva libros.','Alba organiza talleres.','Alba presta revistas.'],'Alba conserva libros, organiza talleres y presta revistas.'))
        self.assertEqual(f['numero_fuentes_complementarias'],3.)
    def test_cobertura_no_veracidad(self):
        f=caracteristicas_b(case(['Alba aprobó la medida. Berta rechazó la medida.'],'Alba rechazó la medida y Berta aprobó la medida.'))
        self.assertEqual(f['cobertura_union'],1.)
        # El clasificador conserva señales de relación y contradicción; esta
        # función jamás convierte cobertura alta en un score de fidelidad.
    def test_distractores(self):
        f=caracteristicas_b(case(['Alba conserva libros.','Berta organiza conciertos.','El banco emite tarjetas.'],'Alba conserva libros.'))
        self.assertEqual(f['ganancia_cobertura_union'],0.)
    def test_duplicadas(self):
        f=caracteristicas_b(case(['Alba conserva libros.']*3,'Alba conserva libros.'))
        self.assertEqual(f['numero_fuentes_complementarias'],0.)
    def test_respuesta_larga_rangos(self):
        f=caracteristicas_b(case(['Alba conserva libros.']*6,'Alba conserva libros. '*12))
        for k in B3[:3]:self.assertTrue(0<=f[k]<=1)


class Sujeto(unittest.TestCase):
    def test_subconjunto_antes_nombre(self):
        f=caracteristicas_b(case(['Solo 3 de los 12 hospitales cerraron consultas externas.'],'Los hospitales cerraron consultas externas.'))
        self.assertEqual(f['sujeto_generalizado'],1.)
    def test_subconjunto_despues_nombre(self):
        f=caracteristicas_b(case(['De los 12 hospitales, 3 cerraron consultas externas.'],'Los hospitales cerraron consultas externas.'))
        self.assertEqual(f['sujeto_generalizado'],1.)
    def test_subconjunto_preservado(self):
        f=caracteristicas_b(case(['3 de los 12 hospitales cerraron consultas externas.'],'3 de los 12 hospitales cerraron consultas externas.'))
        self.assertEqual(f['sujeto_generalizado'],0.)
    def test_grupo_completo_cuantificado(self):
        f=caracteristicas_b(case(['12 de los 12 hospitales cerraron consultas externas.'],'Los hospitales cerraron consultas externas.'))
        self.assertEqual(f['sujeto_generalizado'],0.)
    def test_subconjunto_atributo_distinto(self):
        f=caracteristicas_b(case(['3 de los 12 hospitales cerraron consultas externas.'],'Los hospitales ampliaron laboratorios privados.'))
        self.assertEqual(f['sujeto_generalizado'],0.)
    def test_subconjunto_y_respaldo_completo(self):
        f=caracteristicas_b(case(['3 de los 12 hospitales cerraron consultas externas.','Los hospitales cerraron consultas externas.'],'Los hospitales cerraron consultas externas.'))
        self.assertEqual(f['sujeto_generalizado'],0.)
    def test_generalizacion(self):
        f=caracteristicas_b(case(['Torralcina aprobó la medida.'],'Los municipios de la comarca aprobaron la medida.'))
        self.assertEqual(f['sujeto_generalizado'],1.)
    def test_plural_varias_fuentes(self):
        f=caracteristicas_b(case(['El municipio Alba aprobó la medida.','El municipio Berta aprobó la medida.'],'Los municipios aprobaron la medida.'))
        self.assertEqual(f['sujeto_generalizado'],0.)
    def test_colectivo_explicito(self):
        f=caracteristicas_b(case(['Los municipios de la comarca aprobaron la medida.'],'Los municipios aprobaron la medida.'))
        self.assertEqual(f['sujeto_generalizado'],0.)
    def test_menciona_varios(self):
        f=caracteristicas_b(case(['Alba aprobó la medida.','Berta aprobó la medida.'],'Los municipios Alba y Berta aprobaron la medida.'))
        self.assertEqual(f['sujeto_generalizado'],0.)
    def test_pregunta_plural_no_prueba(self):
        f=caracteristicas_b(case(['Torralcina aprobó la medida.'],'Los municipios aprobaron la medida.','¿Qué municipios aprobaron la medida?'))
        self.assertEqual(f['sujeto_generalizado'],1.)
    def test_restriccion(self):
        f=caracteristicas_b(case(['Torralcina aprobó la medida.'],'Algunos municipios aprobaron la medida.'))
        self.assertEqual(f['sujeto_generalizado'],0.)
    def test_plural_accion_distinta(self):
        f=caracteristicas_b(case(['Torralcina aprobó la medida.'],'Los municipios contrataron trabajadores.'))
        self.assertEqual(f['sujeto_generalizado'],0.)


class Contrato(unittest.TestCase):
    def test_no_ids_etiquetas(self):
        c=case(['El centro solo imparte cursos.'],'El centro imparte cursos.')
        a=caracteristicas_b(c);c.update(id='CAMBIO',tipo='fiel',etiqueta=0)
        self.assertEqual(a,caracteristicas_b(c))
    def test_vacio(self):
        f=caracteristicas_b({})
        self.assertTrue(set(B1+B2+B3+B4)<=set(f))
        self.assertTrue(all(v>=0 for v in f.values()))


if __name__=='__main__':unittest.main()
